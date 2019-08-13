
from kpforecast.ml.n_beats.blocks import NBeatsBlock

import numpy as np
import torch
import torch.nn as nn

class TrendBlock(NBeatsBlock):
    """ Trend N_Beats Block.
    Args:
        -f_b_dim(list/tuple): The integer length of the
             forward and backwards forecast
        -thetas_dim(list/tuple): list or iterable of output
        dimensions of the theta output layers.
        (None by default, results in thetas dim being
        a list same where entries are same as hidden_layer_dim)
        -hidden_layer_dim(int): dimension of input and outputs of
            input and hidden layers (1 by default)
        -num_hidden_layers(int): number of hidden layers
            (2 by default)
        -layer_nonlinearity: torch.nn nonlinearity function
            to use (ReLU by default)
        -layer_w_init: torch.nn.init function to use to
            initialize weight vars 
            (xavier uniform by default)
        -layer_b_init: torch.nn.init function to use to 
            initialize bias constants (zeros by default)
    """
    def __init__(self,
                 f_b_dim,
                 thetas_dim=None,
                 num_hidden_layers=2,
                 hidden_layer_dim=1,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):

        super(TrendBlock, self).__init__(
                    f_b_dim=f_b_dim,
                    thetas_dim=thetas_dim,
                    num_hidden_layers=num_hidden_layers,
                    hidden_layer_dim=hidden_layer_dim,
                    layer_nonlinearity=layer_nonlinearity,
                    layer_w_init=layer_w_init,
                    layer_b_init=layer_b_init)
        self.f_b_dim = f_b_dim
        self.backcast_ls, self.forecast_ls = self.linspace()

    def forward(self, input_val):
        """ Feed Forward function for TrendBlock module.

        Args:
            input val(torch.tensor): input for block
        Returns:
            List of torch.tensors that are outputs of the network,
            Where each output is from a different output head
        """
        thetas = super(TrendBlock, self).forward(input_val)
        backcast = self.trend_model(thetas[1], self.backcast_ls)
        forecast = self.trend_model(thetas[0], self.forecast_ls)
        return forecast, backcast

    def trend_model(self, thetas, t):
        p = thetas.size()[-1]
        assert p <= 4, 'thetas_dim is too big.'
        T = torch.tensor([t ** i for i in range(p)]).float()
        return thetas.mm(T)

    def linspace(self):
        backcast_length = self.f_b_dim[1]
        forecast_length = self.f_b_dim[0]
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length:]
        return b_ls, f_ls
