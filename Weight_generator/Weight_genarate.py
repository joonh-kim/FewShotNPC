import torch
import numpy as np
import random
from data.datamgr import SimpleDataManager
from arguments import parse_args
from model.model import model64, model128, model12
from utils import parse_feature

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    args = parse_args()

    num_model = 50  # which checkpoint file?

    if args.data_set == 'miniimagenet':
        val_file = args.miniimagenet_data_path + '/' + args.split + '.json'
    else:
        val_file = args.path + '/filelists/' + args.data_set + '/' + args.split + '.json'

    image_size = 84

    val_datamgr = SimpleDataManager(image_size, batch_size=args.val_batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = args.path + '/checkpoint/' + args.data_set
    save_file = checkpoint_dir + '/' + args.data_set + '_' + str(num_model) + '.pth'

    if args.backbone == 'Conv64':
        model = model64()
    elif args.backbone == 'Conv128':
        model = model128()
    else:
        model = model12()
    model = model.to(device)
    model.load_state_dict(torch.load(save_file))

    feats = {}
    with torch.no_grad():
        print('Feature extracting...')
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            _, _, base_centers, features = model(x)

            for j in range(len(y)):
                if y[j].item() not in feats.keys():
                    feats[y[j].item()] = {0: features[j].cpu().numpy()}
                else:
                    feats[y[j].item()][len(feats[y[j].item()])] = features[j].cpu().numpy()

            if i % 10 == 9:
                print('Batch {:d}/{:d}'.format(i + 1, len(val_loader)))

        print('Evaluating...')
        acc_all = []
        for i in range(args.iteration):
            class_list = feats.keys()

            select_class = random.sample(class_list, args.n_way)
            z_all = []
            for cl in select_class:
                img_feat = feats[cl]
                perm_ids = np.random.permutation(len(img_feat)).tolist()
                z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(args.n_support + args.n_query)])  # stack each batch

            z_all = torch.from_numpy(np.array(z_all))
            z_all = z_all.to(device)

            """ Evaluation """
            "평균 후 계산"
            # z_support, z_query = parse_feature(z_all, args.n_support)
            #
            # z_support_normalized = z_support / torch.norm(z_support, dim=2).unsqueeze(2)
            # centers = torch.mean(z_support_normalized, dim=1)
            #
            # z_query = z_query.contiguous().view(args.n_way * args.n_query, -1)
            #
            # centers = centers / torch.norm(centers, dim=1).unsqueeze(1)
            # centers_reshape = centers.expand(args.num_class, args.n_way, args.feature_dim)
            # base_centers_reshape = base_centers.unsqueeze(1).expand(args.num_class, args.n_way, args.feature_dim)
            #
            # sim = torch.mul(centers_reshape, base_centers_reshape).sum(dim=2)
            # sim_reshape = sim.unsqueeze(2).expand_as(base_centers_reshape)
            #
            # centers = centers + torch.mul(sim_reshape, base_centers_reshape).sum(dim=0)
            # centers = centers / torch.norm(centers, dim=1).unsqueeze(1)
            #
            # epsilon_reshape = model.classifier.epsilon.unsqueeze(1).expand_as(sim)
            # epsilon = torch.sum(torch.mul(torch.exp(sim), epsilon_reshape), dim=0) / torch.sum(torch.exp(sim), dim=0)
            # epsilon = epsilon.to(device)

            "계산 후 평균"
            z_support, z_query = parse_feature(z_all, args.n_support)

            z_support = z_support.contiguous().view(args.n_way * args.n_support, -1)
            centers = z_support / torch.norm(z_support, dim=1).unsqueeze(1)

            z_query = z_query.contiguous().view(args.n_way * args.n_query, -1)

            centers_reshape = centers.expand(args.num_class, args.n_way * args.n_support, args.feature_dim)
            base_centers_reshape = base_centers.unsqueeze(1).expand(args.num_class, args.n_way * args.n_support, args.feature_dim)

            sim = torch.mul(centers_reshape, base_centers_reshape).sum(dim=2)
            sim_reshape = sim.unsqueeze(2).expand_as(base_centers_reshape)

            centers = centers + torch.mul(sim_reshape, base_centers_reshape).sum(dim=0)
            centers = centers / torch.norm(centers, dim=1).unsqueeze(1)

            centers = centers.view(args.n_way, args.n_support, -1)
            centers = torch.mean(centers, dim=1)
            centers = centers / torch.norm(centers, dim=1).unsqueeze(1)

            """ Score """
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
            y = np.repeat(range(args.n_way), args.n_query)
            acc = np.mean(pred == y) * 100

            acc_all.append(acc)

            if i % 100 == 99:
                print('Iteration {:d}/{:d}'.format(i + 1, args.iteration))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (args.iteration, acc_mean, 1.96 * acc_std / np.sqrt(args.iteration)))

