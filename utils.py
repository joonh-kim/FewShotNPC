import torch
from torch.autograd import Function
import math
from scipy.special import gamma

class Sigma1(Function):
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)  # save input for backward pass

        output = torch.exp(-input) - 1

        return output.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors  # restore input from context

        if ctx.needs_input_grad[0]:
            grad_input = -torch.exp(-input) * grad_output
            grad_input[input > 0] = 0
        return grad_input

class Sigma2(Function):
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)  # save input for backward pass

        output = torch.exp(input) - 1

        return output.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors  # restore input from context

        if ctx.needs_input_grad[0]:
            grad_input = torch.exp(input) * grad_output
            grad_input[input < 0] = 0
        return grad_input

def one_hot_miniimagenet(y, num_class):
    one_hot = torch.zeros(len(y), num_class)
    for i in range(len(y)):
        one_hot[i][y[i]] = 1
    return one_hot

def one_hot_CUB(y, num_class):
    one_hot = torch.zeros(len(y), num_class)
    for i in range(len(y)):
        one_hot[i][y[i] / 2] = 1
        return one_hot

def parse_feature(x, n_support):
    z_all = x
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]

    return z_support, z_query

def NNseparation(n, d):
    if d < 340:
        E = pow(n, -2/(d-1)) * gamma(d/(d-1)) * pow(2 * pow(math.pi, 0.5) * (d-1), 1/(d-1)) * pow(gamma(d/2) / gamma((d-1)/2), -1/(d-1))
    else:
        if d % 2 == 0:
            E = pow(n, -2/(d-1)) * gamma(d/(d-1)) * pow(2 * pow(math.pi, 0.5) * (d-1), 1/(d-1)) * pow(pow((d-2)/2, 0.5), -1/(d-1))
        else:
            E = pow(n, -2/(d-1)) * gamma(d/(d-1)) * pow(2 * pow(math.pi, 0.5) * (d-1), 1/(d-1)) * pow((1/math.e) * pow((d-3)/2, 0.5) * pow((d-1)/(d-3), (d-1)/2), -1/(d-1))
    return E