import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from data.datamgr import SimpleDataManager
from arguments import parse_args
from model.model import model_cc, model128
import random
from utils import one_hot_miniimagenet

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

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    args = parse_args()

    image_size = 84
    val_file = args.miniimagenet_data_path + '/' + args.split + '.json'
    val_datamgr = SimpleDataManager(image_size, batch_size=64)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    save_file_1 = './CC_miniimagenet_400_Conv128.pth'
    save_file_2 = './AF_miniimagenet_450_Conv128.pth'
    save_file_3 = './Ours_miniimagenet_790_Conv128.pth'

    model_1 = model_cc()
    model_2 = model_cc()
    model_3 = model128()

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_3 = model_3.to(device)

    model_1.load_state_dict(torch.load(save_file_1))
    model_2.load_state_dict(torch.load(save_file_2))
    model_3.load_state_dict(torch.load(save_file_3))

    with torch.no_grad():
        feats1 = feature_extract(val_loader, model_1)
        feats2 = feature_extract(val_loader, model_2)
        feats3 = feature_extract(val_loader, model_3)

    feats_np_1 = np.zeros((20, 600, 3200))
    class_list = feats1.keys()
    for cl in class_list:
        for i in range(600):
            feats_np_1[cl - 80][i] = feats1[cl][i]
    feats_np_2 = np.zeros((20, 600, 3200))
    class_list = feats2.keys()
    for cl in class_list:
        for i in range(600):
            feats_np_2[cl - 80][i] = feats2[cl][i]
    feats_np_3 = np.zeros((20, 600, 3200))
    class_list = feats3.keys()
    for cl in class_list:
        for i in range(600):
            feats_np_3[cl - 80][i] = feats3[cl][i]

    cov_1 = np.cov(np.mean(feats_np_1, axis=1)/np.linalg.norm(np.mean(feats_np_1, axis=1), ord=2, axis=1, keepdims=True))
    cov_2 = np.cov(np.mean(feats_np_2, axis=1)/np.linalg.norm(np.mean(feats_np_2, axis=1), ord=2, axis=1, keepdims=True))
    cov_3 = np.cov(np.mean(feats_np_3, axis=1)/np.linalg.norm(np.mean(feats_np_3, axis=1), ord=2, axis=1, keepdims=True))
    cov_list = {0: cov_1, 1: cov_2, 2: cov_3}
    title_list = {0: 'CE', 1: 'AF', 2: 'NPC'}
    y = np.arange(3)

    fig1 = plt.figure(figsize=(9.75, 3))

    grid = ImageGrid(fig1, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 3),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    for ax, cov in zip(grid, y):
        im = ax.imshow(cov_list[cov], interpolation='None')
        ax.set_title(title_list[cov])

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    # plt.show()

    centers_1 = torch.FloatTensor(np.mean(feats_np_1, axis=1)).to(device)
    centers_1 = centers_1 / torch.norm(centers_1, dim=1).unsqueeze(1)
    centers_2 = torch.FloatTensor(np.mean(feats_np_2, axis=1)).to(device)
    centers_2 = centers_2 / torch.norm(centers_2, dim=1).unsqueeze(1)
    centers_3 = torch.FloatTensor(np.mean(feats_np_3, axis=1)).to(device)
    centers_3 = centers_3 / torch.norm(centers_3, dim=1).unsqueeze(1)

    theta_1 = []
    theta_2 = []
    theta_3 = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            features_1 = model_1.feature_extractor(x)
            features_1 = features_1 / torch.norm(features_1, dim=1).unsqueeze(1)
            features_2 = model_2.feature_extractor(x)
            features_2 = features_2 / torch.norm(features_2, dim=1).unsqueeze(1)
            features_3 = model_3.feature_extractor(x)
            features_3 = features_3 / torch.norm(features_3, dim=1).unsqueeze(1)

            y_one_hot = one_hot_miniimagenet(y-80, 20).to(device)
            selected_centers_1 = torch.matmul(y_one_hot, centers_1)
            selected_centers_2 = torch.matmul(y_one_hot, centers_2)
            selected_centers_3 = torch.matmul(y_one_hot, centers_3)

            thetas_1 = torch.acos(torch.mul(features_1, selected_centers_1).sum(dim=1))
            theta_1.append(thetas_1.cpu().numpy().tolist())
            thetas_2 = torch.acos(torch.mul(features_2, selected_centers_2).sum(dim=1))
            theta_2.append(thetas_2.cpu().numpy().tolist())
            thetas_3 = torch.acos(torch.mul(features_3, selected_centers_3).sum(dim=1))
            theta_3.append(thetas_3.cpu().numpy().tolist())

            if i % 10 == 9:
                print('Batch {:d}/{:d}'.format(i + 1, len(val_loader)))

    theta_1 = np.asarray(sum(theta_1, []))
    theta_1 = theta_1 * 180 / math.pi
    theta_2 = np.asarray(sum(theta_2, []))
    theta_2 = theta_2 * 180 / math.pi
    theta_3 = np.asarray(sum(theta_3, []))
    theta_3 = theta_3 * 180 / math.pi

    bins = np.arange(0, 90, 1)

    fig2 = plt.figure()
    plt.hist(theta_1, bins, color='r', edgecolor='black', alpha=0.5, label='CE')
    plt.hist(theta_2, bins, color='g', edgecolor='black', alpha=0.5, label='AF')
    plt.hist(theta_3, bins, color='b', edgecolor='black', alpha=0.5, label='NPC')

    plt.legend()
    plt.xlabel('Angle (degree)')
    plt.ylabel('Number')
    plt.show()
