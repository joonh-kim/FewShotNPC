import torch
import torch.nn as nn
import argparse
from torch import optim
import torch.utils.data as data
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from create_dataset_multinomial import create_dataset


def multinomial_visualization(input, label, input2, label2):
    _fig = plt.figure()
    _plot1 = _fig.add_subplot(1, 2, 1)
    _plot2 = _fig.add_subplot(1, 2, 2)
    for i in range(len(label)):
        if torch.all(label[i] == torch.tensor([1, 0, 0, 0])):
            _plot1.scatter(input[i, 0], input[i, 1], c='y')
        elif torch.all(label[i] == torch.tensor([0, 1, 0, 0])):
            _plot1.scatter(input[i, 0], input[i, 1], c='g')
        elif torch.all(label[i] == torch.tensor([0, 0, 1, 0])):
            _plot1.scatter(input[i, 0], input[i, 1], c='c')
        else:
            _plot1.scatter(input[i, 0], input[i, 1], c='m')
    for i in range(len(label2)):
        if torch.all(label2[i] == torch.tensor([1, 0, 0, 0])):
            _plot2.scatter(input2[i, 0], input2[i, 1], c='y')
        elif torch.all(label2[i] == torch.tensor([0, 1, 0, 0])):
            _plot2.scatter(input2[i, 0], input2[i, 1], c='g')
        elif torch.all(label2[i] == torch.tensor([0, 0, 1, 0])):
            _plot2.scatter(input2[i, 0], input2[i, 1], c='c')
        else:
            _plot2.scatter(input2[i, 0], input2[i, 1], c='m')
    plt.show()

def my_loss(output, target):
    #t1_1 = torch.log(output)
    #t1_2 = t1_1 * target
    t1_1 = output-target
    t1_2 = t1_1*t1_1
    return torch.sum(t1_2)


class eclipse(nn.Module):
    def __init__(self):
        super(eclipse, self).__init__()
        self.rotation_matrix = nn.Parameter(torch.randn(4,2,2))
        self.centers = nn.Parameter(torch.randn(4,2))
        self.radii = nn.Parameter(torch.randn(4, 2))

    def forward(self, input):
        c = self.rotation_matrix.view(8,2)
        d = torch.matmul(input, c.T)
        e = d.view(input.shape[0],4,2)

        f = e-self.centers
        g = f / self.radii
        h = g*g
        i = torch.sum(h,dim=2)
        j = 1-i
        k = torch.sigmoid(j)
        return k


if __name__ == '__main__':

    # " parameters setting "
    # num_data = 1000
    # epochs = 1000
    # alpha = 1  # for sigmoid function
    # lr = 0.001
    # indim = 2
    # numclasses = 4

    torch.manual_seed(7)

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
    training_generator = data.DataLoader(training_set, **params)

    validation_set = create_dataset(train=False, n_samples=args.num_data)
    validation_generator = data.DataLoader(validation_set, **params)

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
        optimizer.zero_grad()
        for inputs, targets in training_generator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # optimizer.zero_grad()
            y_pred = model(inputs)
            loss = my_loss(y_pred.float(), targets.float())
            loss.backward()
            print("Iteration:", epoch, "| loss:", loss.item())
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
                if torch.argmax(train_result[i]) == 0:
                    pred_label.append(torch.tensor([1, 0, 0, 0]))
                elif torch.argmax(train_result[i]) == 1:
                    pred_label.append(torch.tensor([0, 1, 0, 0]))
                elif torch.argmax(train_result[i]) == 2:
                    pred_label.append(torch.tensor([0, 0, 1, 0]))
                else:
                    pred_label.append(torch.tensor([0, 0, 0, 1]))

        for inputs, targets in validation_generator:
            test_data.append(inputs)
            test_label.append(targets)
            test_result = model(inputs)
            for i in range(len(test_result)):
                if torch.argmax(test_result[i]) == 0:
                    pred_test_label.append(torch.tensor([1, 0, 0, 0]))
                elif torch.argmax(test_result[i]) == 1:
                    pred_test_label.append(torch.tensor([0, 1, 0, 0]))
                elif torch.argmax(test_result[i]) == 2:
                    pred_test_label.append(torch.tensor([0, 0, 1, 0]))
                else:
                    pred_test_label.append(torch.tensor([0, 0, 0, 1]))

    train_data = torch.stack(train_data, dim=0); train_data = train_data.view(-1, *train_data.size()[2:])
    train_label = torch.stack(train_label, dim=0); train_label = train_label.view(-1, *train_label.size()[2:])
    pred_label = torch.stack(pred_label); #pred_label = pred_label.view(-1, *pred_label.size()[2:])

    test_data = torch.stack(test_data, dim=0); test_data = test_data.view(-1, *test_data.size()[2:])
    test_label = torch.stack(test_label, dim=0); test_label = test_label.view(-1, *test_label.size()[2:])
    pred_test_label = torch.stack(pred_test_label); #pred_test_label = pred_test_label.view(-1, *pred_test_label.size()[2:])


    train_acc = 100 * torch.all(torch.eq(train_label, pred_label), dim=1).sum() / len(train_label)
    print("Train acc.:", train_acc.item(), "%")

    test_acc = 100 * torch.all(torch.eq(test_label, pred_test_label), dim=1).sum() / len(test_label)
    print("Test acc.:", test_acc.item(), "%")

    # Visualization
    multinomial_visualization(train_data, train_label, test_data, test_label)
    multinomial_visualization(train_data, pred_label, test_data, pred_test_label)

















