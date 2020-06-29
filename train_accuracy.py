import torch
from data.datamgr import SimpleDataManager
from arguments import parse_args
from model.model import *
from utils import *
import os

if __name__ == '__main__':

    args = parse_args()

    for j in range(4):
        num_model = 100 * (j+1)

        # num_model = 400  # which checkpoint file?

        if args.data_set == 'miniimagenet':
            base_file = args.miniimagenet_data_path + '/base.json'
        else:
            base_file = args.path + '/filelists/' + args.data_set + '/base.json'

        image_size = 224

        base_datamgr = SimpleDataManager(image_size, batch_size=args.val_batch_size)
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(save_file))

        correct = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(base_loader):
                x = x.to(device)
                y = y.to(device)

                if args.classifier == 'Ours':
                    _, outputs, _ = model(x)
                elif args.classifier in ['Cosine', 'ArcFace']:
                    outputs = model(x)

                _, predicted = torch.max(outputs, 1)

                if args.data_set == 'miniimagenet':
                    c = (predicted == y).squeeze()
                else:
                    c = (predicted == (y / 2)).squeeze()

                correct += c.sum()

                if i % 100 == 99:
                    print('Batch {:d}/{:d}'.format(i + 1, len(base_loader)))

        print('Accuracy({:d}) {:.2f}%'.format(num_model, correct * 100 / (len(base_loader) * args.base_batch_size)))


