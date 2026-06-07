import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Algo_setuptorch import Params

params = Params()
size = params.size


def activation(x):
    return F.leaky_relu(x, negative_slope=0.01)


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = activation(x)

        return x


# ============================================================
# Main Network
# ============================================================

class DeviationNet(nn.Module):

    def __init__(
        self,
        n_channels,
        hidden=64,
        n_blocks=8,
        n_time_freq=4,
    ):

        super().__init__()

        self.n_channels = n_channels

        self.n_time_freq = n_time_freq
        self.n_time_ch = 2 * n_time_freq

        in_ch = 7 * n_channels + self.n_time_ch

        out_ch = 2 * n_channels

        # ====================================================
        # Learned embedding
        # 1x1 conv mixes heterogeneous channels
        # ====================================================

        self.input_embed = nn.Sequential(

            nn.Conv2d(
                in_ch,
                hidden,
                kernel_size=1
            ),

            nn.InstanceNorm2d(hidden),

            nn.LeakyReLU(0.01)
        )

        # ====================================================
        # Deep residual body
        # ====================================================

        blocks = []

        for _ in range(n_blocks):
            blocks.append(ResidualBlock(hidden))

        self.body = nn.Sequential(*blocks)

        # ====================================================
        # Final projection
        # ====================================================

        self.final = nn.Conv2d(
            hidden,
            out_ch,
            kernel_size=3,
            padding=1
        )
        




    def _time_channels(self, n, B, H, W, device, dtype):
        """Positional encoding of the iteration index `n`, broadcast to
        [B, 2*n_time_freq, H, W]. Frequencies geometric in [1, 1e-4]."""
        if self.n_time_ch == 0:
            return None
        k = torch.arange(self.n_time_freq, device=device, dtype=dtype)
        freqs = torch.exp(-math.log(10000.0) * k / max(self.n_time_freq, 1))
        ang = float(n) * freqs                    
        emb = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(-1)
        return emb.view(1, -1, 1, 1).expand(B, self.n_time_ch, H, W)

    @staticmethod
    def pack(blocks):
        return torch.cat(blocks, dim=1)

    @staticmethod
    def unpack(tensor, shapes):

        out = []

        c = 0

        for s in shapes:

            ch = s[1]

            out.append(tensor[:, c:c+ch])

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
        n=0,
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

       
        time_ch = self._time_channels(
            n, inp.shape[0], inp.shape[-2], inp.shape[-1], inp.device, inp.dtype
        )
        if time_ch is not None:
            inp = torch.cat([inp, time_ch], dim=1)



        h = self.input_embed(inp)
        


        h = self.body(h)



        out = self.final(h)

        B = out.shape[0]

        idx = 0

        u_learned = []
        v_learned = []



        for i in range(2):

            ch = shapes[i][1]

            u_learned.append(
                torch.nan_to_num(
                    out[:, idx:idx+ch]
                )
            )

            idx += ch



        for i in range(2):

            ch = shapes[i][1]

            v_learned.append(
                torch.nan_to_num(
                    out[:, idx:idx+ch]
                )
            )

            idx += ch

        device = out.device

        # spatial size taken from the actual tensors (robust to any image size)
        H, W = out.shape[-2], out.shape[-1]

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