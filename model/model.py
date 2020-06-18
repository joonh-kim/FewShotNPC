import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from arguments import parse_args
from utils import NNseparation
from model.backbone import *

args = parse_args()

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

""" pretrain models """
class cos_softmax(nn.Module):
    def __init__(self, indim, outdim):
        super(cos_softmax, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale_factor = args.scale_factor

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm)

        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm)

        cos_dist = self.L(x_normalized)
        output = self.scale_factor * (cos_dist)

        return output

class model_cc(nn.Module):
    def __init__(self):
        super(model_cc, self).__init__()
        self.feature_extractor = ConvNet128()
        self.classifier = cos_softmax(args.feature_dim, args.num_class)

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.classifier(x)
        return output

class cos_layer(nn.Module):
    def __init__(self):
        super(cos_layer, self).__init__()
        self.centers = nn.Parameter(torch.randn(args.num_class, args.feature_dim).type(torch.cuda.FloatTensor), requires_grad=True)
        # self.epsilon = torch.tensor(NNseparation(args.num_class, args.feature_dim) / 2)
        self.epsilon = torch.tensor(math.pi / 6)
        self.scale_factor = args.scale_factor

    def forward(self, x):
        x_normalized = x / torch.norm(x, dim=1).unsqueeze(1)
        centers_normalized = self.centers / torch.norm(self.centers, dim=1).unsqueeze(1)

        x_reshape = x_normalized.unsqueeze(1).expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])
        centers_reshape = centers_normalized.expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])

        output = torch.mul(x_reshape, centers_reshape).sum(dim=2)
        output1 = output - 1
        output2 = output - torch.cos(self.epsilon)

        return self.scale_factor * output1, self.scale_factor * output2, centers_normalized

class model64(nn.Module):
    def __init__(self):
        super(model64, self).__init__()
        self.feature_extractor = ConvNet64()
        self.classifier = cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2, centers_normalized = self.classifier(x)
        return output1, output2, x

class model128(nn.Module):
    def __init__(self):
        super(model128, self).__init__()
        self.feature_extractor = ConvNet128()
        self.classifier = cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2, centers_normalized = self.classifier(x)
        return output1, output2, x

class model12(nn.Module):
    def __init__(self):
        super(model12, self).__init__()
        self.feature_extractor = MiniImagenetNet()
        self.classifier = cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2, centers_normalized = self.classifier(x)
        return output1, output2, x


""" adaptation models """
class adaptation_cos_softmax(nn.Module):
    def __init__(self, indim, outdim):
        super(adaptation_cos_softmax, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale_factor = args.adaptation_scale_factor

    def forward(self, x):
        centers = torch.mean(x.view(args.n_way, args.n_support, -1), dim=1)

        x_normalized = x / torch.norm(x, dim=1).unsqueeze(1)
        centers_normalized = centers / torch.norm(centers, dim=1).unsqueeze(1)

        x_reshape = x_normalized.unsqueeze(1).expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])
        centers_reshape = centers_normalized.expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])

        output = torch.mul(x_reshape, centers_reshape).sum(dim=2)

        return self.scale_factor * output

class adaptation_model_cc(nn.Module):
    def __init__(self):
        super(adaptation_model_cc, self).__init__()
        self.feature_extractor = ConvNet128()
        self.classifier = adaptation_cos_softmax(args.feature_dim, args.num_class)

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.classifier(x)
        return output

class adaptation_cos_layer(nn.Module):
    def __init__(self):
        super(adaptation_cos_layer, self).__init__()
        self.centers = nn.Parameter(torch.randn(args.num_class, args.feature_dim).type(torch.cuda.FloatTensor), requires_grad=False)
        self.epsilon = torch.tensor(NNseparation(args.n_way, args.feature_dim) / 2)
        self.scale_factor = args.adaptation_scale_factor

    def forward(self, x):
        centers = torch.mean(x.view(args.n_way, args.n_support, -1), dim=1)

        x_normalized = x / torch.norm(x, dim=1).unsqueeze(1)
        centers_normalized = centers / torch.norm(centers, dim=1).unsqueeze(1)

        x_reshape = x_normalized.unsqueeze(1).expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])
        centers_reshape = centers_normalized.expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])

        output = torch.mul(x_reshape, centers_reshape).sum(dim=2)
        output1 = output - 1
        output2 = output - torch.cos(self.epsilon)

        return self.scale_factor * output1, self.scale_factor * output2

class adaptation_model64(nn.Module):
    def __init__(self):
        super(adaptation_model64, self).__init__()
        self.feature_extractor = ConvNet64()
        self.classifier = adaptation_cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2 = self.classifier(x)
        return output1, output2, x

class adaptation_model128(nn.Module):
    def __init__(self):
        super(adaptation_model128, self).__init__()
        self.feature_extractor = ConvNet128()
        self.classifier = adaptation_cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2 = self.classifier(x)
        return output1, output2, x

class adaptation_model12(nn.Module):
    def __init__(self):
        super(adaptation_model12, self).__init__()
        self.feature_extractor = MiniImagenetNet()
        self.classifier = adaptation_cos_layer()

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2 = self.classifier(x)
        return output1, output2, x

""" finetuning models """
class finetuning_cos_softmax(nn.Module):
    def __init__(self, indim, outdim, random_init, weights):
        super(finetuning_cos_softmax, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        if not random_init:
            with torch.no_grad():
                self.L.weight = nn.Parameter(weights, requires_grad=True)
        self.scale_factor = args.adaptation_scale_factor

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm)

        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm)

        cos_dist = self.L(x_normalized)
        output = self.scale_factor * (cos_dist)

        return output

class finetuning_model_cc(nn.Module):
    def __init__(self, tmp, random_init, weights):
        super(finetuning_model_cc, self).__init__()
        self.feature_extractor = ConvNet128()
        with torch.no_grad():
            self.feature_extractor.load_state_dict(tmp)
        self.classifier = finetuning_cos_softmax(args.feature_dim, args.n_way, random_init, weights)

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.classifier(x)
        return output

class finetuning_cos_layer(nn.Module):
    def __init__(self, random_init, centers):
        super(finetuning_cos_layer, self).__init__()
        if random_init:
            self.centers = nn.Parameter(torch.randn(args.n_way, args.feature_dim).type(torch.cuda.FloatTensor), requires_grad=True)
        else:
            self.centers = nn.Parameter(centers, requires_grad=True)
        # self.epsilon = torch.tensor(NNseparation(args.n_way, args.feature_dim) / 2)
        self.epsilon = torch.tensor(math.pi / 3)
        self.scale_factor = args.adaptation_scale_factor

    def forward(self, x):
        x_normalized = x / torch.norm(x, dim=1).unsqueeze(1)
        centers_normalized = self.centers / torch.norm(self.centers, dim=1).unsqueeze(1)

        x_reshape = x_normalized.unsqueeze(1).expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])
        centers_reshape = centers_normalized.expand(x_normalized.shape[0], centers_normalized.shape[0], x_normalized.shape[1])

        output = torch.mul(x_reshape, centers_reshape).sum(dim=2)
        output1 = output - 1
        output2 = output - torch.cos(self.epsilon)

        return self.scale_factor * output1, self.scale_factor * output2, centers_normalized

class finetuning_model(nn.Module):
    def __init__(self, tmp, random_init, centers):
        super(finetuning_model, self).__init__()
        self.feature_extractor = ConvNet128()
        with torch.no_grad():
            self.feature_extractor.load_state_dict(tmp)
        self.classifier = finetuning_cos_layer(random_init, centers)

    def forward(self, x):
        x = self.feature_extractor(x)
        output1, output2, centers_normalized = self.classifier(x)
        return output1, output2, x