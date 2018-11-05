import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy import special


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def calculate_FB_bases(L1):
    maxK = (2 * L1 + 1) ** 2 - 1

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 1.5

    if L1 < 2:
        truncate_freq_factor = 2

    xx, yy = np.meshgrid(range(-L, L + 1), range(-L, L + 1))

    xx = xx / R
    yy = yy / R

    ugrid = np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], 1)
    tgrid, rgrid = cart2pol(ugrid[:, 0], ugrid[:, 1])

    num_grid_points = ugrid.shape[0]

    kmax = 15

    bessel = np.load('bessel.npy')

    B = bessel[(bessel[:, 0] <= kmax) & (bessel[:, 3] <= np.pi * R * truncate_freq_factor)]

    idxB = np.argsort(B[:, 2])

    mu_ns = B[idxB, 2] ** 2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns = np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases = 0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid = rgrid * R_ns[i]

        F = special.jv(ki, r0grid)

        Phi = 1. / np.abs(special.jv(ki + 1, R_ns[i])) * F

        Phi[rgrid >= 1] = 0

        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki, qi, rkqi])
            num_bases = num_bases + 1

        else:
            Psi.append(Phi * np.cos(ki * tgrid) * np.sqrt(2))
            Psi.append(Phi * np.sin(ki * tgrid) * np.sqrt(2))
            kq_Psi.append([ki, qi, rkqi])
            kq_Psi.append([ki, qi, rkqi])
            num_bases = num_bases + 2

    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2 * L + 1, 2 * L + 1).transpose(1, 2, 0)
    psi = p[1:-1, 1:-1, :]

    psi = psi.reshape((2 * L1 + 1) ** 2, num_bases)

    c = np.sqrt(np.sum(psi ** 2, 0).mean())

    psi = psi / c

    return psi


class DCFConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_bases, bases='FB', stride=1, padding=0,
                 dilation=1, bias=True):

        super(DCFConv2d, self).__init__()
        if type(bases) != np.ndarray:
            assert bases in ['FB', 'random', 'PCA']

        if bases == 'FB':
            if kernel_size % 2 == 0:
                raise Exception("kernel_size should be odd")
            base = torch.from_numpy(kernel_size // 2)

            if num_bases > base.shape[1]:
                raise Exception(
                    'The maximum number of bases for kernel size = %d is %d' % (kernel_size, base_np.shape[1]))
            base = base[:, :num_bases]
            base = base.reshape(kernel_size, kernel_size, num_bases)
            base = np.expand_dims(base.transpose(2, 0, 1), 1)
        elif bases == 'random':
            base = np.random.randn(num_bases, 1, kernel_size, kernel_size)
        else:
            assert type(bases) == np.ndarray
            assert bases.shape[2] == kernel_size and bases.shape[3] == kernel_size
            base = bases.reshape(-1, kernel_size * kernel_size).transpose()
            if num_bases > base.shape[1]:
                raise Exception(
                    'The maximum number of bases for the input weight is %d while the number of bases is %d' % (
                    base.shape[1], num_bases))
            transformer = PCA(n_components=num_bases)
            base = transformer.fit_transform(base)
            base = base.transpose().reshape(num_bases, kernel_size, kernel_size)

        self.bases = Parameter(torch.Tensor(base), requires_grad=False)
        self.weight = Parameter(torch.Tensor(out_channels, in_channels * num_bases, 1, 1))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.constant(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):

        in_shape = input.shape()
        input = input.view(-1, 1, in_shape[2], in_shape[3])

        feature = F.conv2d(input, self.bases, stride=self.stride, padding=self.padding, dilation=self.dilation)
        feature = feature.view(in_shape[0], -1, feature.shape[2], feature.shape[3])

        output = F.conv2d(feature, self.weight, bias=self.bias, stride=1, padding=0)

        return output
