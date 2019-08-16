import torch
import torch.nn as nn

from kpforecast.ml.n_beats import Stack, GenericNBeatsBlock

class NBeats(nn.Module):
    """ N-Beats Network.
    N-Beats Network as described by:
    https://arxiv.org/pdf/1905.10437.pdf

    Args:
        - stacks (list/tuple): List/Tuple of the stacks,
            indicated by the block class that each stack
            will be made up of e.g.:
                [GenericNBeatsBlock, GenericNBeatsBlock]
            will result in an Nbeats Architecture with 2
            stacks, and both stacks are made up of GenericNBeats
            Blocks.
            Note: The order of this list/tuple will indicate the order
            of the stacks

    Args For Stack:
        - f_b_dim(list/tuple): The integer length of the
                forward and backwards forecast
        - block_cls(function): object class that
            inherits from NBeatsBlock object. Dictates
            what type of block to be used in stack.
            NBeatsBlock can be found in
                kpforecast.ml.n_beats.blocks
        - num_blocks_per_stack(int):number of blocks in a stack
            (5 by default)
        - share_stack_weights(bool): True if blocks within
            stack should share weights, False otherwise.
            (False by default)

    Args For Block:
        - hidden_layer_dim(int): dimension of input and outputs of
            input and hidden layers (1 by default)
        - thetas_dim(list/tuple): list or iterable of output
            dimensions of the theta output layers.
            (None by default, results in thetas dim being
            a list same where entries are same as hidden_layer_dim)
        - num_hidden_layers(int): number of hidden layers
            (2 by default)
        - layer_nonlinearity: torch.nn nonlinearity function
            to use.
            (ReLU by default)
        - layer_w_init: torch.nn.init function to use to
            initialize weight vars.
            (xavier uniform by default)
        - layer_b_init: torch.nn.init function to use to 
            initialize bias constants. 
            (zeros by default)
    """

    def __init__(self,
                 stacks,
                 f_b_dim,
                 num_blocks_per_stack=[5],
                 share_stack_weights=False,
                 thetas_dims=None,
                 shared_g_theta=False,
                 hidden_layer_dim=1,
                 num_hidden_layers=2,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):
        
        assert(len(thetas_dims) == len(stacks) == len(num_blocks_per_stack))
        self._stack_classes = stacks
        self._f_b_dim = f_b_dim
        self._num_blocks_per_stack = num_blocks_per_stack
        self._share_stack_weights = share_stack_weights
        self._thetas_dims = thetas_dims
        self._shared_g_theta = shared_g_theta
        self._hidden_layer_dim = hidden_layer_dim
        self._num_hidden_layers = num_hidden_layers
        self._layer_nonlinearity = layer_nonlinearity
        self._layer_w_init = layer_w_init
        self._layer_b_init = layer_b_init
        super().__init__()
        self._stacks = nn.ModuleList()
        for idx, block_cls in enumerate(self._stack_classes):
            new_stack = Stack(f_b_dim=self._f_b_dim,
                              block_cls=block_cls,
                              num_blocks=self._num_blocks_per_stack[idx],
                              share_stack_weights=self._share_stack_weights,
                              thetas_dim=self._thetas_dims[idx],
                              shared_g_theta=self._shared_g_theta,
                              hidden_layer_dim=self._hidden_layer_dim,
                              num_hidden_layers=self._num_hidden_layers,
                              layer_nonlinearity=self._layer_nonlinearity,
                              layer_w_init=self._layer_w_init,
                              layer_b_init=self._layer_b_init)
            self._stacks.append(new_stack)

    def forward(self, input_var):
        forecast_length = self._f_b_dim[0]
        forecasted_values = torch.zeros((forecast_length))
        # import ipdb; ipdb.set_trace()
        residuals = input_var
        for idx, stack in enumerate(self._stacks):
            local_stack_forecast, local_stack_backcast = stack(residuals)
            forecasted_values = forecasted_values + (local_stack_forecast)
            residuals = residuals - (local_stack_backcast)
        return forecasted_values, residuals
