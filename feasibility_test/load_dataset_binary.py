import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
from matplotlib import pyplot as plt


class load_dataset(data.Dataset):
    def __init__(self, train):
        super(load_dataset, self).__init__()

        if train:
            self.inputs = torch.load('train_data.pt')
            self.targets = torch.load('train_label.pt')
        else:
            self.inputs = torch.load('test_data.pt')
            self.targets = torch.load('test_label.pt')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
       return self.inputs[index], self.targets[index]



if __name__ == '__main__':
    np.random.seed(0)
    n_samples = 1000
    data = torch.randn(n_samples, 2)
    label = torch.zeros((data.shape[0]))
    for i in range(n_samples):
        if data[i, 0] < data[i, 1]:
            label[i] = 1
    #plt.figure()
    fig1 = plt.figure()
    train_label_plot = fig1.add_subplot(1, 1, 1)

    for i in range(len(label)):
        if label[i] == 0:
            train_label_plot.scatter(data[i, 0], data[i, 1], c='b')
        else:
            train_label_plot.scatter(data[i, 0], data[i, 1], c='g')
    #            plt.show()
    torch.save(data, 'train_data.pt')
    torch.save(label, 'train_label.pt')


    test_data = torch.randn(n_samples, 2)
    test_label = torch.zeros(n_samples)
    for i in range(n_samples):
        if test_data[i, 0] < test_data[i, 1]:
            test_label[i] = 1

    fig2 = plt.figure()
    test_label_plot = fig2.add_subplot(1, 1, 1)
    for i in range(len(test_label)):
        if test_label[i] == 0:
            test_label_plot.scatter(test_data[i, 0], test_data[i, 1], c='b')
        else:
            test_label_plot.scatter(test_data[i, 0], test_data[i, 1], c='g')
    #           plt.show()
    torch.save(test_data, 'test_data.pt')
    torch.save(test_label, 'test_label.pt')
