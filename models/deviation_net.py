import torch
import torch.nn as nn
import torch.nn.functional as F


class DeviationNet(nn.Module):
    """
    Small convolutional network (PyTorch port of the `convnet` of Banert et al.):

        InstanceNorm -> [Conv3x3 (dilated) -> InstanceNorm -> LeakyReLU] * 5 -> Conv3x3

The dilations (1, 2, 4, 8, 1) give a receptive field of about 35x35 pixels.

    It reads the primal blocks (u, w) of seven state quantities and predicts
    the primal blocks of the two deviation directions. The dual blocks (p, q)
    of the deviations are set to zero.
    """

    def __init__(self, n_channels, hidden=32, dilations=(1, 2, 4, 8, 1)):
        super().__init__()
        self.n_channels = n_channels          # channels of (u, w) = 3
        self.n_layers = len(dilations)

        self.in_norm = nn.InstanceNorm2d(7 * n_channels, affine=True)

        # stored flat as conv, norm, conv, norm, ...
        layers = []
        ch = 7 * n_channels
        for d in dilations:
            layers.append(nn.Conv2d(ch, hidden, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.InstanceNorm2d(hidden, affine=True))
            ch = hidden
        self.layers = nn.ModuleList(layers)

        self.final = nn.Conv2d(hidden, 2 * n_channels, kernel_size=3, padding=1)

    def forward(self, x, p, y, z, u_prev, v_prev, Cy):
        primal = [b for blocks in (x, p, y, z, u_prev, v_prev, Cy) for b in blocks[:2]]

        h = self.in_norm(torch.cat(primal, dim=1))
        for i in range(self.n_layers):
            h = self.layers[2 * i](h)
            h = self.layers[2 * i + 1](h)
            h = F.leaky_relu(h, inplace=True)
        out = self.final(h)

        # output channels: (u-block, w-block) of the first direction, then of the second
        ch_u = x[0].shape[1]
        n = self.n_channels
        zeros = [torch.zeros_like(x[2]), torch.zeros_like(x[3])]
        u_raw = [out[:, :ch_u], out[:, ch_u:n]] + zeros
        v_raw = [out[:, n:n + ch_u], out[:, n + ch_u:]] + zeros
        return u_raw, v_raw
