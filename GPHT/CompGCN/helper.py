import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

np.set_printoptions(precision=4)


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.fft.irfft(com_mult(torch.fft.rfft(a, 1), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
