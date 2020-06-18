import os
import time
import torch
import torch.nn as nn
import argparse
from torch import optim
import torch.utils.data as data
import random
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from load_dataset_binary import load_dataset
from create_dataset_binary import create_dataset

def binary_visualization(input, label, input2=None, label2=None):
    if label2 is not None:
        _fig = plt.figure()
        _plot1 = _fig.add_subplot(1, 2, 1)
        _plot2 = _fig.add_subplot(1, 2, 2)
        for i in range(len(label)):
            if label[i] == 0:
                _plot1.scatter(input[i, 0], input[i, 1], c='b')
            else:
                _plot1.scatter(input[i, 0], input[i, 1], c='g')
        for i in range(len(label2)):
            if label2[i] == 0:
                _plot2.scatter(input2[i, 0], input2[i, 1], c='b')
            else:
                _plot2.scatter(input2[i, 0], input2[i, 1], c='g')
        plt.show()
    else:
        fig = plt.figure()
        _plot = fig.add_subplot(1, 1, 1)
        for i in range(len(input)):
            if label[i] == 0:
                _plot.scatter(input[i, 0], input[i, 1], c='b')
            else:
                _plot.scatter(input[i, 0], input[i, 1], c='g')
        plt.show()

def my_loss(output, target):
    t1_1 = torch.log(output)
    t1_2 = -target * t1_1

    t2_1 = torch.log(1-output)
    t2_2 = -(1-target)
    t2_3 = t2_1*t2_2
    loss = t1_2 + t2_3

    return torch.sum(loss)
    #loss = -target*torch.log(output)-(1-target)*torch.log(1-output)
    #return torch.sum(loss)


class eclipse(nn.Module):
    def __init__(self):
        super(eclipse, self).__init__()

        self.R_1 = nn.Parameter(torch.randn(2)); self.R_2 = nn.Parameter(torch.randn(2))
        self.r_1 = nn.Parameter(torch.randn(1)); self.r_2 = nn.Parameter(torch.randn(1))
        self.c_1 = nn.Parameter(torch.randn(1)); self.c_2 = nn.Parameter(torch.randn(1))

    def forward(self, input):
        t1_1 = torch.matmul(input, self.R_1)
        t1_2 = t1_1 - self.c_1
        t1_3 = t1_2 / self.r_1
        t2_1 = torch.matmul(input, self.R_2)
        #t2_1 = t2_1.view(t2_1.shape[0])
        t2_2 = t2_1 - self.c_2
        t2_3 = t2_2 / self.r_2
        forward_result = torch.sigmoid(1-t1_3*t1_3-t2_3*t2_3)
        return forward_result


if __name__ == '__main__':

    # data generation
    np.random.seed(0)
    # Parameters
    parser = argparse.ArgumentParser(description='Autograd Basics')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    args = parser.parse_args()

    # Generators
    params = {'batch_size': 100, 'shuffle': True, 'num_workers': 0}
    training_set = create_dataset(train=True, n_samples=args.num_data)
    #training_set = load_dataset(train=True)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = create_dataset(train=False, n_samples=args.num_data)
    #validation_set = load_dataset(train=False)
    validation_generator = data.DataLoader(validation_set, **params)

    #loss_fn = torch.nn.MSELoss(reduction='mean')
    #loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = my_loss()

    model = eclipse()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-2)

    if torch.cuda.device_count() > 1:
        if args.gpu_ids == None:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            device = torch.device('cuda:0')
        else:
            print("Let's use", len(args.gpu_ids), "GPUs!")
            device = torch.device('cuda:' + str(args.gpu_ids[0]))
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('args.gpu_ids', args.gpu_ids)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        for inputs, targets in training_generator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = my_loss(y_pred.float(), targets.float())
            loss.backward()
        optimizer.step()

    train_data = []; train_label = []; test_data = []; test_label = []
    pred_label = []; pred_test_label = []

    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, targets in training_generator:
            train_data.append(inputs)
            train_label.append(targets)
            train_result = model(inputs)
            for i in range(len(train_result)):
                if train_result[i] > 0.5:
                    pred_label.append(1)
                else:
                    pred_label.append(0)

        for inputs, targets in validation_generator:
            test_data.append(inputs)
            test_label.append(targets)
            test_result = model(inputs)
            for i in range(len(test_result)):
                if test_result[i] > 0.5:
                    pred_test_label.append(1)
                else:
                    pred_test_label.append(0)

    train_data = torch.stack(train_data, dim=0); train_data = train_data.view(-1, *train_data.size()[2:])
    train_label = torch.stack(train_label, dim=0); train_label = train_label.view(-1, *train_label.size()[2:])
    pred_label = torch.tensor(pred_label); pred_label = pred_label.view(-1, *pred_label.size()[2:])

    test_data = torch.stack(test_data, dim=0); test_data = test_data.view(-1, *test_data.size()[2:])
    test_label = torch.stack(test_label, dim=0); test_label = test_label.view(-1, *test_label.size()[2:])
    pred_test_label = torch.tensor(pred_test_label); pred_test_label = pred_test_label.view(-1, *pred_test_label.size()[2:])

    train_acc = 100 * (train_label == pred_label).sum() / len(train_label)
    print("Train acc.:", train_acc.item(), "%")

    test_acc = 100 * (test_label == pred_test_label).sum() / len(test_label)
    print("Test acc.:", test_acc.item(), "%")


    # Visualization
    binary_visualization(train_data, train_label, test_data, test_label)
    binary_visualization(train_data, pred_label, test_data, pred_test_label)
