import torch
import torch.optim as optim
import os
from arguments import parse_args
from data.datamgr import SimpleDataManager
from model.model import *
from utils import *

def criterion(output1, output2, target, batch_size):
    s1 = Sigma1.apply
    s2 = Sigma2.apply
    alpha = 1/15

    loss = (target * s1(output1) + alpha * (1 - target) * s2(output2)).sum(dim=1)
    loss = loss.sum() / batch_size

    return loss

if __name__ == '__main__':

    args = parse_args()

    checkpoint_dir = args.path + '/checkpoint/' + args.data_set
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if args.data_set == 'miniimagenet':
        base_file = args.miniimagenet_data_path + '/base.json'
    else:
        base_file = args.path + '/filelists/' + args.data_set + '/base.json'

    if args.num_epochs == -1:
        if args.data_set == 'miniimagenet':
            if args.bacbone == 'ResNet18':
                num_epochs = 400
            elif args.bacbone == 'Conv128':
                num_epochs = 800
            else:
                raise NotImplementedError("Unavailable backbone architecture")
        elif args.data_set == 'CUB':
            num_epochs = 5000
        else:
            raise NotImplementedError("Unavailable dataset")

    if args.backbone == 'ResNet18':
        image_size = 224
    elif args.bacbone == 'Conv128':
        image_size = 84
    else:
        raise NotImplementedError("Unavailable backbone architecture")

    base_datamgr = SimpleDataManager(image_size, batch_size=args.base_batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.classifier == 'Ours':
        if args.backbone == 'Conv128':
            model = model128()
        elif args.backbone == 'ResNet18':
            model = model18()
        else:
            raise NotImplementedError("Unavailable backbone architecture")
    elif args.classifier in ['CC', 'AF', 'CF']:
        model = model_cc()

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    CE_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(base_loader):
            x = x.to(device)
            if args.classifier == 'NPC':
                if args.data_set == 'miniimagenet':
                    y = one_hot_miniimagenet(y, args.num_class).type(torch.cuda.FloatTensor)
                else:
                    y = one_hot_CUB(y, args.num_class).type(torch.cuda.FloatTensor)
            y = y.to(device)

            optimizer.zero_grad()

            if args.classifier == 'NPC':
                output1, output2, _ = model(x)
                loss = criterion(output1, output2, y, args.base_batch_size)
            elif args.classifier in ['CC', 'AF', 'CF']:
                output = model(x)
                if args.classifier == 'AF':
                    y_one_hot = one_hot_miniimagenet(y, args.num_class).type(torch.cuda.FloatTensor)
                    output_target = torch.mul(torch.cos(torch.acos(torch.mul(output/args.scale_factor, y_one_hot))
                                                        + args.margin), y_one_hot)
                    output = args.scale_factor * (output_target + torch.mul(output/args.scale_factor, 1-y_one_hot))
                if args.classifier == 'CF':
                    y_one_hot = one_hot_miniimagenet(y, args.num_class).type(torch.cuda.FloatTensor)
                    output_target = torch.mul(output / args.scale_factor, y_one_hot) - args.margin
                    output = args.scale_factor * (output_target + torch.mul(output/args.scale_factor, 1-y_one_hot))
                loss = CE_loss(output, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch + 1, i + 1, len(base_loader),
                                                                        running_loss / float(i + 1)))
                running_loss = 0.0

        if epoch == 0 or epoch % 100 == 99:
            save_file = os.path.join(checkpoint_dir, args.data_set + '_' + str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_file)

    """ Test accuracy evaluation """
    for j in range(int(num_epochs / 100)):
        num_model = 100 * (j + 1)

        checkpoint_dir = args.path + '/checkpoint/' + args.data_set
        save_file = os.path.join(checkpoint_dir, args.data_set + '_' + str(num_model) + '.pth')

        if args.classifier == 'NPC':
            if args.backbone == 'Conv128':
                model = model128()
            elif args.backbone == 'ResNet18':
                model = model18()
            else:
                raise NotImplementedError
        elif args.classifier in ['CC', 'AF', 'CF']:
            model = model_cc()

        model = model.to(device)

        loaded_params = torch.load(save_file)
        new_params = model.state_dict().copy()
        for i in loaded_params:
            i_parts = i.split('.')
            if i_parts[0] == 'module':
                new_params['.'.join(i_parts[1:])] = loaded_params[i]
            else:
                new_params[i] = loaded_params[i]
        model.load_state_dict(new_params)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        correct = 0.0
        with torch.no_grad():
            for k, (x, y) in enumerate(base_loader):
                x = x.to(device)
                y = y.to(device)

                if args.classifier == 'NPC':
                    _, outputs, _ = model(x)
                elif args.classifier in ['CC', 'AF', 'CF']:
                    outputs = model(x)

                _, predicted = torch.max(outputs, 1)

                if args.data_set == 'miniimagenet':
                    c = (predicted == y).squeeze()
                else:
                    c = (predicted == (y/2)).squeeze()

                correct += c.sum()

        print('Accuracy({:d}) {:.2f}%'.format(num_model, correct * 100 / (len(base_loader) * args.base_batch_size)))