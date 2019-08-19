
from kpforecast.ml.n_beats.blocks import NBeatsBlock
from torch.nn import functional as F

import numpy as np
import torch
import torch.nn as nn

class TrendBlock(NBeatsBlock):
    def __init__(self,
                f_b_dim,
                thetas_dim=(2,2),
                num_hidden_layers=3,
                hidden_layer_dim=8,
                layer_nonlinearity=nn.ReLU,
                layer_w_init=nn.init.xavier_uniform_,
                layer_b_init=nn.init.zeros_,
                shared_g_theta=None):
        super(TrendBlock, self).__init__(f_b_dim,
                                         thetas_dim,
                                         num_hidden_layers,
                                         hidden_layer_dim)
        self.backcast_linspace, self.forecast_linspace = self.linspace()

    def forward(self, x):
        def trend_model(thetas, t):
            p = thetas.size()[-1]
            assert p <= 4, 'thetas_dim is too big.'
            T = torch.tensor([t ** i for i in range(p)]).float()
            ret = thetas.mm(T)
            return ret
        thetas = super(TrendBlock, self).forward(x)
        backcast = trend_model(thetas[1], self.backcast_linspace)
        forecast = trend_model(thetas[0], self.forecast_linspace)
        return forecast, backcast

    def linspace(self):
        backcast_length = self._f_b_dim[1]
        forecast_length = self._f_b_dim[0]
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length:]
        return b_ls, f_ls
    