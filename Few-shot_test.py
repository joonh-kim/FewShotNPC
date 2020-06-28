import torch
import numpy as np
import random
from data.datamgr import SimpleDataManager
from arguments import parse_args
from model.model import *
from utils import parse_feature
import os

def feature_extract(val_loader, model):
    feats = {}
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)

        features = model.feature_extractor(x)

        for j in range(len(y)):
            if y[j].item() not in feats.keys():
                feats[y[j].item()] = {0: features[j].cpu().numpy()}
            else:
                feats[y[j].item()][len(feats[y[j].item()])] = features[j].cpu().numpy()

        if i % 50 == 49:
            print('Batch {:d}/{:d}'.format(i + 1, len(val_loader)))

    return feats

def evaluate(feats, n_way, n_support, n_query):
    class_list = feats.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = feats[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])

    z_all = torch.from_numpy(np.array(z_all))
    z_all = z_all.to(device)

    """ Evaluation """
    z_support, z_query = parse_feature(z_all, n_support)

    z_support_normalized = z_support / torch.norm(z_support, dim=2).unsqueeze(2)
    centers = torch.mean(z_support_normalized, dim=1)

    z_query = z_query.contiguous().view(n_way * n_query, -1)

    z_query_normalized = z_query / torch.norm(z_query, dim=1).unsqueeze(1)
    centers_normalized = centers / torch.norm(centers, dim=1).unsqueeze(1)

    z_query_normalized_reshape = z_query_normalized.unsqueeze(1).expand(z_query_normalized.shape[0],
                                                                        centers_normalized.shape[0],
                                                                        z_query_normalized.shape[1])
    centers_normalized_reshape = centers_normalized.expand(z_query_normalized.shape[0],
                                                           centers_normalized.shape[0],
                                                           z_query_normalized.shape[1])

    scores = torch.mul(z_query_normalized_reshape, centers_normalized_reshape).sum(dim=2)

    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100

    return acc

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    args = parse_args()

    if args.data_set == 'miniimagenet':
        val_file = args.miniimagenet_data_path + '/' + args.split + '.json'
    else:
        val_file = args.path + '/filelists/' + args.data_set + '/' + args.split + '.json'

    if args.num_epochs == -1:
        if args.data_set == 'miniimagenet':
            num_epochs = 800
        else:
            num_epochs = 5000

    image_size = 224

    val_datamgr = SimpleDataManager(image_size, batch_size=args.val_batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i in range(int(num_epochs / 100)):
        num_model = 100 * (i + 1)
        # num_model = 50

        checkpoint_dir = args.path + '/checkpoint/' + args.data_set
        save_file = checkpoint_dir + '/' + args.data_set + '_' + str(num_model) + '.pth'

        if args.classifier == 'Ours':
            if args.backbone == 'Conv64':
                model = model64()
            elif args.backbone == 'Conv128':
                model = model128()
            elif args.backbone == 'ResNet12':
                model = model12()
            elif args.backbone == 'ResNet18':
                model = model18()
            else:
                raise NotImplementedError
        elif args.classifier in ['Cosine', 'ArcFace']:
            model = model_cc()
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(save_file))

        with torch.no_grad():
            print('Feature extracting...')
            feats = feature_extract(val_loader, model)

            print('Evaluating...')
            acc_all = []
            for j in range(args.iteration):
                acc = evaluate(feats, args.n_way, args.n_support, args.n_query)
                acc_all.append(acc)

            acc_all = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)

            print('%d Test Acc(%d) = %4.2f%% +- %4.2f%%' % (args.iteration, num_model, acc_mean,
                                                            1.96 * acc_std / np.sqrt(args.iteration)))




