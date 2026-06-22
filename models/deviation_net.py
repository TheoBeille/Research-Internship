import torch
import torch.nn as nn
import torch.nn.functional as F


def activation(x):
    return F.leaky_relu(x, negative_slope=0.01)


# ============================================================
# Simple feed-forward conv net  (paper 1 style)
# ============================================================
#
# Direct PyTorch port of the reference paper's TensorFlow `convnet`:
#
#     x = inst_norm(input)
#     for _ in range(n_layers):
#         x = conv(x, filters=32, k=3, SAME)
#         x = inst_norm(x)
#         x = leaky_relu(x)
#     out = conv(x, filters=out_ch, k=3, SAME)
#
# No residual blocks and no 1x1 channel-mixing embedding (the previous
# DeviationNet had 8 residual blocks at 64 channels). Fewer/lighter layers
# means far fewer activations to keep for backprop -> much lower training
# memory, which is what lets us push the image size up to 512x512.
#
# The forward interface (inputs/outputs) is kept identical to the old network
# so `UnrolledFBS` does not need any change.

class DeviationNet(nn.Module):

    def __init__(
        self,
        n_channels,
        hidden=32,
        n_blocks=8,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_layers = n_blocks

        # 7 stacked state blocks, each using only the first 2 sub-blocks (u, w).
        in_ch = 7 * n_channels
        out_ch = 2 * n_channels

        # inst_norm on the raw input (as in the paper's convnet).
        self.in_norm = nn.InstanceNorm2d(in_ch, affine=True)

        # Plain conv stack: [conv -> inst_norm] * n_layers, leaky_relu applied
        # in forward. Stored flat as conv, norm, conv, norm, ...
        layers = []
        ch = in_ch
        for _ in range(n_blocks):
            layers.append(nn.Conv2d(ch, hidden, kernel_size=3, padding=1))
            layers.append(nn.InstanceNorm2d(hidden, affine=True))
            ch = hidden
        self.layers = nn.ModuleList(layers)

        # Final projection to the deviation channels.
        self.final = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)

    @staticmethod
    def pack(blocks):
        return torch.cat(blocks, dim=1)

    @staticmethod
    def unpack(tensor, shapes):
        out = []
        c = 0
        for s in shapes:
            ch = s[1]
            out.append(tensor[:, c:c + ch])
            c += ch
        return out

    def forward(
        self,
        shapes,
        x_blocks,
        p_blocks,
        y_blocks,
        z_blocks,
        u_prev,
        v_prev,
        Cy,
    ):

        inputs = [
            self.pack(x_blocks[:2]),
            self.pack(p_blocks[:2]),
            self.pack(y_blocks[:2]),
            self.pack(z_blocks[:2]),
            self.pack(u_prev[:2]),
            self.pack(v_prev[:2]),
            self.pack(Cy[:2]),
        ]

        inp = torch.cat(inputs, dim=1)

        h = self.in_norm(inp)

        for i in range(self.n_layers):
            conv = self.layers[2 * i]
            norm = self.layers[2 * i + 1]
            h = conv(h)
            h = norm(h)
            h = activation(h)

        out = self.final(h)

        B = out.shape[0]
        H, W = out.shape[-2], out.shape[-1]
        idx = 0

        u_learned = []
        v_learned = []

        for i in range(2):
            ch = shapes[i][1]
            u_learned.append(torch.nan_to_num(out[:, idx:idx + ch]))
            idx += ch

        for i in range(2):
            ch = shapes[i][1]
            v_learned.append(torch.nan_to_num(out[:, idx:idx + ch]))
            idx += ch

        device = out.device

        u_zeros = [
            torch.zeros(B, shapes[2][1], H, W, device=device),
            torch.zeros(B, shapes[3][1], H, W, device=device),
        ]

        v_zeros = [
            torch.zeros(B, shapes[2][1], H, W, device=device),
            torch.zeros(B, shapes[3][1], H, W, device=device),
        ]

        u_raw = u_learned + u_zeros
        v_raw = v_learned + v_zeros

        return u_raw, v_raw
