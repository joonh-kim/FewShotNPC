import torch
import torch.optim as optim
import os
from arguments import parse_args
from data.datamgr import SetDataManager
from model.model import *
from utils import *
import numpy as np

def criterion(output1, output2, target, batch_size):
    s1 = Sigma1.apply
    s2 = Sigma2.apply
    alpha = 1/4

    loss = (target * s1(output1) + alpha * (1 - target) * s2(output2)).sum()

    return loss / batch_size

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
    model_num = 400

    args = parse_args()

    checkpoint_dir = os.path.join(args.path, 'checkpoint', args.data_set)

    if args.data_set == 'miniimagenet':
        loadfile = args.miniimagenet_data_path + '/' + args.split + '.json'
    else:
        loadfile = args.path + '/filelists/' + args.data_set + '/' + args.split + '.json'
    modelfile = os.path.join(checkpoint_dir, args.data_set + '_' + str(model_num) + ".pth")

    image_size = 84

    datamgr = SetDataManager(image_size, args.n_way, args.n_support, args.n_query, args.iteration)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.classifier == 'Ours':
        if args.backbone == 'Conv64':
            model = adaptation_model64()
        elif args.backbone == 'Conv128':
            model = adaptation_model128()
        else:
            model = adaptation_model12()
    elif args.classifier in ['Cosine', 'ArcFace']:
        model = adaptation_model_cc()

    # model = model.to(device)
    # tmp = torch.load(modelfile)
    # model.load_state_dict(tmp)
    #
    # optimizer = optim.Adam(model.parameters(), lr=0.0006)  #lr=0.0006
    # CE_loss = nn.CrossEntropyLoss()

    acc_all = [[] for k in range(args.adaptation_step + 1)]
    for i, (x, _) in enumerate(data_loader):
        x_support, x_query = parse_support_and_query(x, args.n_way, args.n_support, args.n_query)
        x_support = x_support.to(device)
        x_query = x_query.to(device)

        if args.classifier == 'Ours':
            y_support = one_hot_miniimagenet(np.repeat(range(args.n_way), args.n_support), args.n_way)
            y_support = y_support.to(device)
        elif args.classifier in ['Cosine', 'ArcFace']:
            y_support = np.repeat(range(args.n_way), args.n_support)
            y_support = torch.tensor(y_support).to(device)

        model = model.to(device)
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp)

        if args.adaptation_step == 0:
            with torch.no_grad():
                acc = evaluate(model, x_support, x_query, args.n_way, args.n_support)
                acc_all[0].append(acc)

            if i % 50 == 49:
                print('Iteration {:d}/{:d}'.format(i + 1, len(data_loader)))

        else:
            optimizer = optim.Adam(model.parameters(), lr=args.adaptation_lr)  # lr=0.0006
            CE_loss = nn.CrossEntropyLoss()

            for j in range(args.adaptation_step):
                optimizer.zero_grad()
                if args.classifier == 'Ours':
                    output1, output2, _ = model(x_support)
                    loss = criterion(output1, output2, y_support, args.n_way * args.n_support)
                elif args.classifier in ['Cosine', 'ArcFace']:
                    output = model(x_support)
                    if args.classifier == 'ArcFace':
                        y_one_hot = one_hot_miniimagenet(y_support, args.n_way).type(torch.cuda.FloatTensor)
                        output_target = torch.mul(torch.cos(torch.acos(torch.mul(output / args.scale_factor, y_one_hot)) + args.margin), y_one_hot)
                        output = args.scale_factor * (output_target + torch.mul(output / args.scale_factor, 1 - y_one_hot))
                    loss = CE_loss(output, y_support)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    acc = evaluate(model, x_support, x_query, args.n_way, args.n_support)
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

