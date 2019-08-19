import torch
import torch.nn as nn


class Stack(nn.Module):
    """ Stack in N-Beats Network.
    Stacks are residual networks, made up of multiple blocks
    and their architecture can be easily understood in Figure 1 
    of the N-Beats paper.

    Args:
        - block_cls(function): object class that
            inherits from NBeatsBlock object. Dictates
            what type of block to be used in stack.
            NBeatsBlock can be found in
                kpforecast.ml.n_beats.blocks
        - num_blocks(int):number of blocks in a stack
            (5 by default)
        - share_stack_weights(bool): True if block within
            stack should share weights, False otherwise.
            (False by default)

    Args For Block:
        - f_b_dim(list/tuple): The integer length of the
            forward and backwards forecast
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
                 f_b_dim,
                 block_cls,
                 num_blocks=5,
                 share_stack_weights=False,
                 thetas_dim=None,
                 shared_g_theta=False,
                 hidden_layer_dim=1,
                 num_hidden_layers=2,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):
        super().__init__()
        self._f_b_dim = f_b_dim
        self._block_cls = block_cls
        self._num_blocks = num_blocks
        self._share_stack_weights = share_stack_weights
        self._hidden_layer_dim = hidden_layer_dim
        self._thetas_dim = thetas_dim
        self._shared_g_theta = shared_g_theta
        self._num_hidden_layers = num_hidden_layers
        self._layer_nonlinearity = layer_nonlinearity
        self._layer_w_init = layer_w_init
        self._layer_b_init = layer_b_init

        # blocks that make up the stack
        self._blocks = nn.ModuleList()
        num_blocks_to_create = self._num_blocks

        # In the case that the stack should
        # have blocks share weights, we will take
        # the output of the block, do any necessary
        # operations on that output, and then feed that
        # back into the block, to emulate blocks that
        # share weights
        block_module = self._block_cls(
                f_b_dim=self._f_b_dim,
                thetas_dim=self._thetas_dim,
                shared_g_theta=self._shared_g_theta,
                hidden_layer_dim=self._hidden_layer_dim,
                num_hidden_layers=self._num_hidden_layers,
                layer_nonlinearity=self._layer_nonlinearity,
                layer_w_init=self._layer_w_init,
                layer_b_init=self._layer_b_init)
        self._blocks.append(block_module)
        for i in range(1, num_blocks_to_create):
            if not self._share_stack_weights:
                block_module = self._block_cls(
                f_b_dim=self._f_b_dim,
                thetas_dim=self._thetas_dim,
                shared_g_theta=self._shared_g_theta,
                hidden_layer_dim=self._hidden_layer_dim,
                num_hidden_layers=self._num_hidden_layers,
                layer_nonlinearity=self._layer_nonlinearity,
                layer_w_init=self._layer_w_init,
                layer_b_init=self._layer_b_init)
            self._blocks.append(block_module)

    def forward(self, input_val):
        """feed forward method for stack modules.
        
        args:
            - input_val(torch.tensor): input value to stack
        returns:
            - residual values (backcasted values - input_val).
                These can be thought of as the values that the
                stack couldn't fit to.
            - forecasted values. These are the forecasts based on 
                the input_val.
        """
        forecast_length, backcast_length = self._f_b_dim[0], self._f_b_dim[1]
        forecasted_values = torch.zeros((forecast_length))
        residual_values = input_val
        for block in self._blocks:
            local_block_forecast, local_block_backcast = block(residual_values)
            forecasted_values = forecasted_values + local_block_forecast
            residual_values = residual_values - (local_block_backcast)
        return forecasted_values, residual_values
