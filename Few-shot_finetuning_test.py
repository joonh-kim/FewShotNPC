import torch
import torch.optim as optim
import os
from arguments import parse_args
from data.datamgr import SetDataManager
from model.model import *
from model.backbone import *
from utils import *
import numpy as np

def criterion(output1, output2, target, batch_size):
    s1 = Sigma1.apply
    s2 = Sigma2.apply
    alpha = 1/15

    loss = (target * s1(output1) + alpha * (1 - target) * s2(output2)).sum(dim=1)
    loss = loss.sum() / batch_size

    return loss

def parse_support_and_query(x, n_way, n_support, n_query):
    x_support = x[:, :n_support]
    x_support = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:])

    x_query = x[:, args.n_support:]
    x_query = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])

    return x_support, x_query

def evaluate(model, x_support, x_query, n_way, n_support):
    support_feat = model.feature_extractor(x_support)
    support_feat = support_feat.contiguous().view(n_way, n_support, -1)
    centers = torch.mean(support_feat, dim=1)

    query_feat = model.feature_extractor(x_query)
    query_feat_normalized = query_feat / torch.norm(query_feat, dim=1).unsqueeze(1)
    centers_normalized = centers / torch.norm(centers, dim=1).unsqueeze(1)

    query_feat_reshape = query_feat_normalized.unsqueeze(1).expand(query_feat_normalized.shape[0],
                                                                   centers_normalized.shape[0],
                                                                   query_feat_normalized.shape[1])
    centers_reshape = centers_normalized.expand(query_feat_normalized.shape[0],
                                                centers_normalized.shape[0],
                                                query_feat_normalized.shape[1])

    score = torch.mul(query_feat_reshape, centers_reshape).sum(dim=2)
    pred = score.data.cpu().numpy().argmax(axis=1)
    y_query = np.repeat(range(args.n_way), args.n_query)
    acc = np.mean(pred == y_query) * 100

    return acc

if __name__ == '__main__':
    model_num = 600

    random_init = False

    args = parse_args()

    checkpoint_dir = os.path.join(args.path, 'checkpoint', args.data_set)

    if args.data_set == 'miniimagenet':
        loadfile = args.miniimagenet_data_path + '/' + args.split + '.json'
    else:
        loadfile = args.path + '/filelists/' + args.data_set + '/' + args.split + '.json'
    modelfile = os.path.join(checkpoint_dir, args.data_set + '_' + str(model_num) + ".pth")
    # modelfile = './checkpoint/miniimagenet/Rebuttal/ResNet18_ep110_batch128_multigpu/miniimagenet_400.pth'

    if args.backbone == 'ResNet18':
        image_size = 224
    elif args.bacbone == 'Conv128':
        image_size = 84
    else:
        raise NotImplementedError("Unavailable backbone architecture")

    datamgr = SetDataManager(image_size, args.n_way, args.n_support, args.n_query, args.iteration)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tmp = torch.load(modelfile)
    state_keys = list(tmp.keys())
    for i, key in enumerate(state_keys):
        if "feature_extractor." in key:
            if "module" not in key:
                newkey = key.replace("feature_extractor.", "")
            else:
                newkey = key.replace("module.feature_extractor.", "")
            tmp[newkey] = tmp.pop(key)
        else:
            tmp.pop(key)

    if args.backbone == 'Conv128':
        backbone = ConvNet128()
    elif args.backbone == 'ResNet18':
        backbone = ResNet18()
    else:
        raise NotImplementedError("Unavailable backbone architecture")
    backbone = backbone.to(device)
    backbone.load_state_dict(tmp)

    acc_all = [[] for k in range(args.adaptation_step + 1)]
    for i, (x, _) in enumerate(data_loader):
        x_support, x_query = parse_support_and_query(x, args.n_way, args.n_support, args.n_query)
        x_support = x_support.to(device)
        x_query = x_query.to(device)

        y_support = np.repeat(range(args.n_way), args.n_support)
        y_support = torch.tensor(y_support).to(device)

        support_feat = backbone(x_support)
        support_feat = support_feat.contiguous().view(args.n_way, args.n_support, -1)
        centers = torch.mean(support_feat, dim=1)

        model = finetuning_model_cc(tmp, random_init, centers)
        model.to(device)

        if args.adaptation_step == 0:
            with torch.no_grad():
                score = model(x_query)
                pred = score.data.cpu().numpy().argmax(axis=1)
                y_query = np.repeat(range(args.n_way), args.n_query)
                acc = np.mean(pred == y_query) * 100
                acc_all[0].append(acc)

            if i % 50 == 49:
                print('Iteration {:d}/{:d}'.format(i + 1, len(data_loader)))

        else:
            optimizer = optim.Adam(model.parameters(), lr=args.adaptation_lr)  #lr=0.0005
            CE_loss = nn.CrossEntropyLoss()

            for j in range(args.adaptation_step):
                optimizer.zero_grad()
                if args.classifier == 'NPC':
                    output = model(x_support)

                elif args.classifier in ['CC', 'AF', 'CF']:
                    output = model(x_support)
                    if args.classifier == 'AF':
                        y_one_hot = one_hot_miniimagenet(y_support, args.n_way).type(torch.cuda.FloatTensor)
                        output_target = torch.mul(
                            torch.cos(torch.acos(torch.mul(output / args.scale_factor, y_one_hot)) + args.margin),
                            y_one_hot)
                        output = args.scale_factor * (output_target + torch.mul(output / args.scale_factor, 1 - y_one_hot))
                    if args.classifier == 'CF':
                        y_one_hot = one_hot_miniimagenet(y_support, args.n_way).type(torch.cuda.FloatTensor)
                        output_target = torch.mul(output / args.scale_factor, y_one_hot) - args.margin
                        output = args.scale_factor * (
                                    output_target + torch.mul(output / args.scale_factor, 1 - y_one_hot))
                loss = CE_loss(output, y_support)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    score = model(x_query)
                    pred = score.data.cpu().numpy().argmax(axis=1)
                    y_query = np.repeat(range(args.n_way), args.n_query)
                    acc = np.mean(pred == y_query) * 100
                    acc_all[j + 1].append(acc)

            if i % 50 == 49:
                print('Iteration {:d}/{:d}'.format(i + 1, len(data_loader)))

    if args.adaptation_step == 0:
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all, 1)
        acc_std = np.std(acc_all, 1)
        print('Step 0 | Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean[0], 1.96 * acc_std[0] / np.sqrt(args.iteration)))
    else:
        acc_all = acc_all[1:]
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all, 1)
        acc_std = np.std(acc_all, 1)
        for l in range(args.adaptation_step):
            print('Step %d | Test Acc = %4.2f%% +- %4.2f%%' % (l + 1, acc_mean[l], 1.96 * acc_std[l] / np.sqrt(args.iteration)))



