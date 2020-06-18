import torch
import torch.nn as nn
import argparse
from torch import optim
import torch.utils.data as data
from matplotlib import pyplot as plt
from feasibility_test.create_dataset_multinomial import create_dataset
from custom_functions import Sigma1, Sigma2
from tensorboardX import SummaryWriter

summary = SummaryWriter('multinomial_circle_autograd/epoch_1000_minibatch_100')

def multinomial_visualization(input, label, input2, label2, model):
    _fig = plt.figure()
    _plot1 = _fig.add_subplot(1, 2, 1)
    _plot2 = _fig.add_subplot(1, 2, 2)

    circle1_train = plt.Circle((model.centers[0, 0], model.centers[0, 1]), model.radii[0], color='y', alpha=0.5)
    circle2_train = plt.Circle((model.centers[1, 0], model.centers[1, 1]), model.radii[1], color='g', alpha=0.5)
    circle3_train = plt.Circle((model.centers[2, 0], model.centers[2, 1]), model.radii[2], color='c', alpha=0.5)
    circle4_train = plt.Circle((model.centers[3, 0], model.centers[3, 1]), model.radii[3], color='m', alpha=0.5)
    circle1_test = plt.Circle((model.centers[0, 0], model.centers[0, 1]), model.radii[0], color='y', alpha=0.5)
    circle2_test = plt.Circle((model.centers[1, 0], model.centers[1, 1]), model.radii[1], color='g', alpha=0.5)
    circle3_test = plt.Circle((model.centers[2, 0], model.centers[2, 1]), model.radii[2], color='c', alpha=0.5)
    circle4_test = plt.Circle((model.centers[3, 0], model.centers[3, 1]), model.radii[3], color='m', alpha=0.5)

    _plot1.add_artist(circle1_train)
    _plot1.add_artist(circle2_train)
    _plot1.add_artist(circle3_train)
    _plot1.add_artist(circle4_train)
    _plot2.add_artist(circle1_test)
    _plot2.add_artist(circle2_test)
    _plot2.add_artist(circle3_test)
    _plot2.add_artist(circle4_test)

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

def my_loss(output, target, dist):
    s1 = Sigma1.apply
    s2 = Sigma2.apply

    loss = -(target * torch.log(s1(output)) + (1 - target) * torch.log(s2(output)) + target * torch.log(torch.exp(-dist))).sum()

    return loss

class hypersphere(nn.Module):
    def __init__(self):
        super(hypersphere, self).__init__()
        self.centers = nn.Parameter(torch.randn(4, 2).type(torch.cuda.FloatTensor), requires_grad=True)
        # self.radii = nn.Parameter(torch.rand(4).type(torch.cuda.FloatTensor), requires_grad=True)
        self.radii = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor), requires_grad=True)

    def forward(self, input):
        input_reshape = input.unsqueeze(1).expand(input.shape[0], self.centers.shape[0], input.shape[1])
        centers_reshape = self.centers.expand(input.shape[0], self.centers.shape[0], input.shape[1])
        # output = self.radii.pow(2) - (input - self.centers).pow(2).sum(dim=1) # only for batch_size=1
        output = self.radii - (input_reshape - centers_reshape).pow(2).sum(dim=2).sqrt()
        return output

    def distance(self, input):
        input_reshape = input.unsqueeze(1).expand(input.shape[0], self.centers.shape[0], input.shape[1])
        centers_reshape = self.centers.expand(input.shape[0], self.centers.shape[0], input.shape[1])
        # output = (input - self.centers).pow(2).sum(dim=1) # only for batch_size=1
        output = (input_reshape - centers_reshape).pow(2).sum(dim=2).sqrt()
        return output

if __name__ == '__main__':

    torch.manual_seed(3)

    # Parameters
    parser = argparse.ArgumentParser(description='Autograd Basics')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    # Generators
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    training_set = create_dataset(train=True, n_samples=args.num_data)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = create_dataset(train=False, n_samples=args.num_data)
    validation_generator = data.DataLoader(validation_set, **params)

    model = hypersphere()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-2)

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

    # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)

    for epoch in range(args.num_epochs):
        avg_loss = 0
        model.train()

        optimizer.zero_grad()
        for inputs, targets in training_generator:
            # optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            y_pred = model(inputs)
            dist = model.distance(inputs)
            loss = my_loss(y_pred, targets, dist)

            avg_loss += loss

            loss.backward()
            # optimizer.step()
        optimizer.step()
        print("Epoch:", epoch, "| loss:", (avg_loss / args.num_data).item())
        # # Tensorboard
        # if epoch % 10 == 0:
        #     summary.add_scalar('loss', (avg_loss / args.num_data).item(), epoch)


    train_data = []; train_label = []; test_data = []; test_label = []
    pred_label = []; pred_test_label = []

    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, targets in training_generator:
            train_data.append(inputs)
            train_label.append(targets)

            inputs = inputs.to(device)
            targets = targets.to(device)

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

            inputs = inputs.to(device)
            targets = targets.to(device)

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
    # multinomial_visualization(train_data, train_label, test_data, test_label, model)
    multinomial_visualization(train_data, pred_label, test_data, pred_test_label, model)
