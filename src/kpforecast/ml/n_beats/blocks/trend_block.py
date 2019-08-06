
from kpforecast.ml.n_beats.blocks import NBeatsBlock

import torch
import torch.nn as nn

class TrendBlock(nn.Module):

    def __init__(self):
        pass

    def forward(self, input_val):
        pass


    def trend_model(self, thetas, t):
        p = thetas.size()[-1]
        assert p <= 4, 'thetas_dim is too big.'
        T = torch.tensor([t ** i for i in range(p)]).float()
        return thetas.mm(T)
