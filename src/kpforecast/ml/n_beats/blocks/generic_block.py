import torch.nn as nn

from kpforecast.ml.n_beats.blocks.block import NBeatsBlock

class GenericNBeatsBlock(NBeatsBlock):
    """ Generic N_Beats Block.
    Args:
        -f_b_dim(list/tuple): The integer length of the
             forward and backwards forecast
        -hidden_layer_dim(int): dimension of input and outputs of
            input and hidden layers (1 by default)
        -thetas_dim(list/tuple): list or iterable of output
            dimensions of the theta output layers.
            (None by default, results in thetas dim being
             a list same where entries are same as hidden_layer_dim)
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
                 shared_g_theta=False,
                 hidden_layer_dim=1,
                 num_hidden_layers=2,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):

        self._shared_g_theta = shared_g_theta
        if(shared_g_theta):
            if(len(thetas_dim) == 2):
                if (not (thetas_dim[0] == thetas_dim[1])):
                    raise Exception("When sharing g theta in generic block. thetas_dim[0] and thetas_dim[1] must be equal")
            if not f_b_dim[0] == f_b_dim[1]:
                raise Exception("When sharing g theta in generic block, the length of the forecast and backcast must be the same")
        super(GenericNBeatsBlock, self).__init__(
                       f_b_dim=f_b_dim,
                       thetas_dim=thetas_dim,
                       num_hidden_layers=num_hidden_layers,
                       hidden_layer_dim=hidden_layer_dim,
                       layer_nonlinearity=layer_nonlinearity,
                       layer_w_init=layer_w_init,
                       layer_b_init=layer_b_init)

        self._g_theta_out_layer = nn.ModuleList()
        for i in range(len(self._thetas_dim)):
            out_layer = nn.Sequential()
            linear_layer = nn.Linear(thetas_dim[i], f_b_dim[i])
            layer_w_init(linear_layer.weight)
            layer_b_init(linear_layer.bias)
            out_layer.add_module("g_theta_" + str(i), linear_layer)
            if layer_nonlinearity:
                out_layer.add_module('non_linearity', layer_nonlinearity())
            self._g_theta_out_layer.append(out_layer)

    def forward(self, input_val):
        """ Feed Forward function for GenericNBeatsBlock
            module.
        Args:
            input val(torch.tensor): input for
                GenericNBeatsBlock
        Returns:
            List of torch.tensors that are outputs
            of the network, Where each output is from
            a different output head
        """
        thetas = super(GenericNBeatsBlock, self).forward(input_val)
        if self._shared_g_theta:
            return [self._g_theta_out_layer[0](theta) for theta in thetas]
        else:
            if len(thetas) != len(self._g_theta_out_layer):
                raise Exception ("number of theta output heads must be \
                                  must be the same as num of g's (\
                                  function generators)")
            return [layer(theta) for (layer, theta)
                    in zip(self._g_theta_out_layer, thetas)]
