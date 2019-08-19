from kpforecast.ml.n_beats.blocks import NBeatsBlock

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

class SeasonalityBlock(NBeatsBlock):

    def __init__(self,
                    f_b_dim,
                    thetas_dim=(8,8),
                    num_hidden_layers=2,
                    hidden_layer_dim=1,
                    layer_nonlinearity=nn.ReLU,
                    layer_w_init=nn.init.xavier_uniform_,
                    layer_b_init=nn.init.zeros_,
                    shared_g_theta=None):

        super(SeasonalityBlock, self).__init__(f_b_dim,
                                                thetas_dim=thetas_dim,
                                                hidden_layer_dim=hidden_layer_dim)
        self.backcast_linspace, self.forecast_linspace = self.linspace()

    def forward(self, x):
        def seasonality_model(self, thetas, t):
            p = thetas.size()[-1]
            assert p < 10, 'thetas_dim is too big.'
            p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
            s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
            s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
            S = torch.cat([s1, s2])
            return thetas.mm(S)
        thetas = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self, thetas[1], self.backcast_linspace)
        forecast = seasonality_model(self, thetas[0], self.forecast_linspace)
        return forecast, backcast

    def linspace(self):
        backcast_length = self._f_b_dim[1]
        forecast_length = self._f_b_dim[0]
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length:]
        return b_ls, f_ls