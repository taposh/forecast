from kpforecast.ml.n_beats.blocks import NBeatsBlock

import torch
import torch.nn as nn

class SeasonalityBlock(nn.Module):

    def __init__(self):
        pass

    def forward(self, input_val):
        pass

    def seasonality_model(self, thetas, t):
        p = thetas.size()[-1]
        assert p < 10, 'thetas_dim is too big.'
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
        s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
        S = torch.cat([s1, s2])
        return thetas.mm(S)