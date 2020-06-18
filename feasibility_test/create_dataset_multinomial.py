import torch.utils.data
import torch
import torch.utils.data as data



class create_dataset(data.Dataset):
    def __init__(self, train, n_samples):
        super(create_dataset, self).__init__()

        torch.manual_seed(3)

        if train:
            data = torch.randn(n_samples, 2)
            #label = torch.zeros(n_samples)
            label = [[] for i in range(data.shape[0])]
            for i in range(n_samples):
                if data[i, 0] >= 0 and data[i, 1] >= 0:
                    label[i] = [1, 0, 0, 0]
                elif data[i, 0] >= 0 and data[i, 1] < 0:
                    label[i] = [0, 1, 0, 0]
                elif data[i, 0] < 0 and data[i, 1] >= 0:
                    label[i] = [0, 0, 1, 0]
                else:
                    label[i] = [0, 0, 0, 1]
            self.inputs = data
            self.targets = torch.tensor(label)
        else:
            test_data = torch.randn(n_samples, 2)
            #test_label = torch.zeros(n_samples)

            test_label = [[] for i in range(n_samples)]
            for i in range(n_samples):
                if test_data[i, 0] >= 0 and test_data[i, 1] >= 0:
                    test_label[i] = [1, 0, 0, 0]
                elif test_data[i, 0] >= 0 and test_data[i, 1] < 0:
                    test_label[i] = [0, 1, 0, 0]
                elif test_data[i, 0] < 0 and test_data[i, 1] >= 0:
                    test_label[i] = [0, 0, 1, 0]
                else:
                    test_label[i] = [0, 0, 0, 1]
            self.inputs = test_data
            self.targets = torch.tensor(test_label)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
       return self.inputs[index], self.targets[index]



