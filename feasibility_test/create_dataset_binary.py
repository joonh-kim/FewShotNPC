import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
from matplotlib import pyplot as plt


class create_dataset(data.Dataset):
    def __init__(self, train, n_samples):
        super(create_dataset, self).__init__()
        np.random.seed(0)

        if train:
            data = torch.randn(n_samples, 2)
            label = torch.zeros((data.shape[0]))
            for i in range(n_samples):
                if data[i, 0] < data[i, 1]:
                    label[i] = 1
            self.inputs = data
            self.targets = label
        else:
            test_data = torch.randn(n_samples, 2)
            test_label = torch.zeros(n_samples)
            for i in range(n_samples):
                if test_data[i, 0] < test_data[i, 1]:
                    test_label[i] = 1
            self.inputs = test_data
            self.targets = test_label

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
       return self.inputs[index], self.targets[index]


