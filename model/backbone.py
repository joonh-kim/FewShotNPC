from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        ('bn', nn.BatchNorm2d(out_channels, momentum=1)),
        # ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5,
              momentum=0.1):
    # momentum = 1 restricts stats to the current mini-batch
    # This hack only works when momentum is 1 and avoids needing to track
    # running stats by substituting dummy variables
    size = int(np.prod(np.array(input.data.size()[1])))
    running_mean = torch.zeros(size).cuda()
    running_var = torch.ones(size).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)


class OmniglotNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(OmniglotNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', conv_block(x_dim, hid_dim)),
            ('block2', conv_block(hid_dim, hid_dim)),
            ('block3', conv_block(hid_dim, hid_dim)),
            ('block4', conv_block(hid_dim, z_dim)),
        ]))

    def forward(self, x, weights=None):
        if weights is None:
            x = self.encoder(x)
        else:
            x = F.conv2d(x, weights['encoder.block1.conv.weight'], weights['encoder.block1.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block1.bn.weight'], bias=weights['encoder.block1.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block2.conv.weight'], weights['encoder.block2.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block2.bn.weight'], bias=weights['encoder.block2.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block3.conv.weight'], weights['encoder.block3.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block3.bn.weight'], bias=weights['encoder.block3.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block4.conv.weight'], weights['encoder.block4.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block4.bn.weight'], bias=weights['encoder.block4.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        return x.view(x.size(0), -1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, filters, pool_padding=0):
        super(ResBlock, self).__init__()
        self.conv1 = conv(in_channels, filters)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = conv(filters, filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = conv(filters, filters)
        self.bn3 = nn.BatchNorm2d(filters)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = conv(in_channels, filters, kernel_size=1, padding=0)

        self.maxpool = nn.MaxPool2d(2, padding=pool_padding)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        residual = self.conv4(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out += residual
        out = self.maxpool(out)
        out = self.dropout(out)

        return out


class MiniImagenetNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''

    def __init__(self, in_channels=3):
        super(MiniImagenetNet, self).__init__()
        self.block1 = ResBlock(in_channels, 64)
        self.block2 = ResBlock(64, 96)
        self.block3 = ResBlock(96, 128, pool_padding=1)
        self.block4 = ResBlock(128, 256, pool_padding=1)
        self.conv1 = conv(256, 2048, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.conv2 = conv(2048, 384, kernel_size=1, padding=0)

    def forward(self, x, weights=None):
        if weights is None:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)
        else:
            raise ValueError('Not implemented yet')
        return x.view(x.size(0), -1)

class ConvNet64(nn.Module):
    def __init__(self):
        super(ConvNet64, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.shape[0], -1)
        return x

class ConvNet128(nn.Module):
    def __init__(self):
        super(ConvNet128, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.shape[0], -1)
        return x

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)