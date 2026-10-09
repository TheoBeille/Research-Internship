import random

import torch.nn as nn

from algorithm.fbs_step import unroll
from models.deviation_net import DeviationNet


class UnrolledFBS(nn.Module):
    """
    Forward-backward splitting unrolled for T iterations, where the
    deviations are predicted by a network and rescaled by the safeguard.
    """

    def __init__(self, params, shapes, n_channels, T=10, alpha=0.99, random_horizon=True,
                 use_safeguard=True):
        super().__init__()
        self.params = params
        self.shapes = shapes
        self.T = T
        self.alpha = alpha
        # train on T + randint(0, T) iterations (as in Banert et al.) instead of T
        self.random_horizon = random_horizon
        # False: the network output is applied directly (no convergence guarantee)
        self.use_safeguard = use_safeguard
        self.dev_net = DeviationNet(n_channels)

    def forward(self, functions, T=None, use_safeguard=None, keep_history=False,
                progress=False):
        """Returns (kkt, residuals, x_hist), see algorithm.fbs_step.unroll.
        Outside training the deviations stop after T iterations and the
        classical scheme restarts from p_T."""
        if T is None:
            T = self.T
            if self.training and self.random_horizon:
                T += random.randint(0, self.T)

        if use_safeguard is None:
            use_safeguard = self.use_safeguard

        device = next(self.parameters()).device
        return unroll(functions, self.params, self.shapes, T, device,
                      direction=self.dev_net, alpha=self.alpha,
                      use_safeguard=use_safeguard, keep_history=keep_history,
                      progress=progress, restart=None if self.training else self.T)
