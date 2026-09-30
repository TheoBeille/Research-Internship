import torch
import torch.nn as nn
from algorithm.fbs_step import one_step
from algorithm.normalization import block_norm_sq
import random
from models.deviation_net import DeviationNet
from torch.utils.checkpoint import checkpoint


class UnrolledFBS(nn.Module):
    """
    Unrolled forward-backward splitting model with learned deviations 
    and gradient checkpointing to save VRAM at clinical resolution (512x512).
    """

    def __init__(self, params, shapes, n_channels, T=5, net_hidden=32, net_blocks=2, alpha=0.99):
        super().__init__()

        self.params = params
        self.shapes = shapes
        self.n_channels = n_channels
        self.n_blocks = len(shapes)
        self.T = T
        self.alpha = alpha

        self.dev_net = DeviationNet(n_channels=n_channels, hidden=net_hidden, n_blocks=net_blocks)

    def _init_state(self, init_state):
        dev = init_state.device
        B = init_state.shape[0]
        x = [
            torch.zeros((B, *self.shapes[0][1:]), device=dev, requires_grad=True),
            torch.zeros((B, *self.shapes[1][1:]), device=dev, requires_grad=True),
            torch.zeros((B, *self.shapes[2][1:]), device=dev, requires_grad=True),
            torch.zeros((B, *self.shapes[3][1:]), device=dev, requires_grad=True),
        ]

        y_prev = [t.clone() for t in x]
        p_prev = [t.clone() for t in x]
        z_prev = [t.clone() for t in x]

        u = [torch.zeros_like(t) for t in x]
        v = [torch.zeros_like(t) for t in x]
        u_prev = [t.clone() for t in u]
        v_prev = [t.clone() for t in v]
        
        return x, y_prev, p_prev, z_prev, u, v, u_prev, v_prev

    def _checkpointed_step(self, n, C, RA, compute_delta, 
                           x0, x1, x2, x3, 
                           yp0, yp1, yp2, yp3, 
                           pp0, pp1, pp2, pp3, 
                           zp0, zp1, zp2, zp3, 
                           u0, u1, u2, u3, 
                           v0, v1, v2, v3, 
                           up0, up1, up2, up3, 
                           vp0, vp1, vp2, vp3):
        """
        Helper function packaging lists into flat tensors for checkpointing.
        """
        x = [x0, x1, x2, x3]
        y_prev = [yp0, yp1, yp2, yp3]
        p_prev = [pp0, pp1, pp2, pp3]
        z_prev = [zp0, zp1, zp2, zp3]
        u = [u0, u1, u2, u3]
        v = [v0, v1, v2, v3]
        u_prev = [up0, up1, up2, up3]
        v_prev = [vp0, vp1, vp2, vp3]

        x_new, y, p, z, res = one_step(
            x=x, y_prev=y_prev, p_prev=p_prev, z_prev=z_prev,
            u=u, v=v, n=n, params=self.params, C=C, RA=RA,
        )

        delta = compute_delta(p, x, p_prev, z, z_prev, y, y_prev, u, v, n)

        x_new = [t.float() for t in x_new]
        p = [t.float() for t in p]
        y = [t.float() for t in y]
        z = [t.float() for t in z]

        Cy = C(y)

        u_raw, v_raw = self.dev_net(
            shapes=self.shapes,
            x_blocks=x_new,
            p_blocks=p,
            y_blocks=y,
            z_blocks=z,
            u_prev=u_prev,
            v_prev=v_prev,
            Cy=Cy,
        )

        u_raw_norm = block_norm_sq(u_raw).sqrt().clamp(min=1e-6)
        v_raw_norm = block_norm_sq(v_raw).sqrt().clamp(min=1e-6)

        u_raw = [u_i / u_raw_norm for u_i in u_raw]
        v_raw = [v_i / v_raw_norm for v_i in v_raw]

        params = self.params
        lam = float(params.lam(n + 1))
        mu = float(params.mu(n + 1))
        lpm = lam + mu

        theta_hat = float(params.theta_hat(n + 1))
        theta = float(params.theta(n + 1))
        theta_tilde = float(params.theta_tilde(n + 1))

        c_u = lpm * theta_tilde / theta_hat
        c_v = lpm * theta_hat / theta

        norm_u_sq = block_norm_sq(u_raw)
        norm_v_sq = block_norm_sq(v_raw)

        Q = c_u * norm_u_sq + c_v * norm_v_sq
        budget = float(params.zeta) * (delta.clamp(min=0.0))
        ratio = torch.sqrt(budget / Q)
        scale = self.alpha * ratio

        next_u = [scale * u_i for u_i in u_raw]
        next_v = [scale * v_i for v_i in v_raw]

        # Flatten output to return individual tensors for checkpoint compatibility
        return (
            x_new[0], x_new[1], x_new[2], x_new[3],
            y[0], y[1], y[2], y[3],
            p[0], p[1], p[2], p[3],
            z[0], z[1], z[2], z[3],
            next_u[0], next_u[1], next_u[2], next_u[3],
            next_v[0], next_v[1], next_v[2], next_v[3],
            u[0], u[1], u[2], u[3],       # becomes new u_prev
            v[0], v[1], v[2], v[3],       # becomes new v_prev
            res
        )

    def forward(self, initial_state, functions, return_all=False):
        C = functions["C"]
        RA = functions["RA"]
        compute_delta = functions["compute_delta_torch"]

        x, y_prev, p_prev, z_prev, u, v, u_prev, v_prev = self._init_state(initial_state)

        residuals = []
        AxCx = []
        objectives = []

        if return_all:
            x_hist, y_hist, p_hist, z_hist = [], [], [], []
            u_hist, v_hist, delta_hist = [], [], []
            
        T_run = self.T + random.randint(0, self.T)

        for n in range(T_run):
            # Appel du checkpointing PyTorch pour recalculer les activations à la volée pendant le backward
            out = checkpoint(
                self._checkpointed_step,
                n, C, RA, compute_delta,
                x[0], x[1], x[2], x[3],
                y_prev[0], y_prev[1], y_prev[2], y_prev[3],
                p_prev[0], p_prev[1], p_prev[2], p_prev[3],
                z_prev[0], z_prev[1], z_prev[2], z_prev[3],
                u[0], u[1], u[2], u[3],
                v[0], v[1], v[2], v[3],
                u_prev[0], u_prev[1], u_prev[2], u_prev[3],
                v_prev[0], v_prev[1], v_prev[2], v_prev[3],
                use_reentrant=False
            )

            # Reconstitution des listes d'états à partir de la sortie plate du checkpoint
            x = list(out[0:4])
            y_prev = list(out[4:8])
            p_prev = list(out[8:12])
            z_prev = list(out[12:16])
            u = list(out[16:20])
            v = list(out[20:24])
            u_prev = list(out[24:28])
            v_prev = list(out[28:32])
            res = out[32]

            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res)

            AxCx.append(functions['kkt_residual_norm'](x))
            objectives.append(functions['objective'](x))

            if return_all:
                x_hist.append([t.clone() for t in x])
                y_hist.append([t.clone() for t in y_prev])
                p_hist.append([t.clone() for t in p_prev])
                z_hist.append([t.clone() for t in z_prev])
                u_hist.append([t.clone() for t in u])
                v_hist.append([t.clone() for t in v])

        if return_all:
            history = {
                "x": x_hist, "y": y_hist, "p": p_hist, "z": z_hist,
                "u": u_hist, "v": v_hist,
            }
            return AxCx, residuals, objectives, history

        return AxCx, residuals, objectives, x[0]