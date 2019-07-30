import torch.nn as nn

from kpforecast.ml.n_beats import NBeatsBlock

class GenericNBeatsBlock(NBeatsBlock):
    """ Generic N_Beats Block.
    Args:
        -f_b_dim(list/tuple): The integer length of the
             forward and backwards forecast
        -layer_dim(int): dimension of input and outputs of
            input and hidden layers (1 by default)
        -thetas_dim(list/tuple): list or iterable of output
            dimensions of the theta output layers.
            (None by default, results in thetas dim being
             a list same where entries are same as layer_dim)
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
                 layer_dim=1,
                 thetas_dim=None,
                 shared_g_theta=False,
                 num_hidden_layers=2,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):

        self._shared_g_theta = shared_g_theta
        if(shared_g_theta):
            if(len(layer_dim) > 1):
                raise Exception("When sharing Theta outputs, \
                                 thetas_dim must be a tuple/list of size 1")

        super(GenericNBeatsBlock, self).__init__(
                       layer_dim=layer_dim,
                       thetas_dim=thetas_dim,
                       num_hidden_layers=num_hidden_layers,
                       layer_nonlinearity=layer_nonlinearity,
                       layer_w_init=layer_w_init,
                       layer_b_init=layer_b_init)

        self._g_theta_out_layer = nn.ModuleList()
        for i in range(len(f_b_dim)):
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
            of the network,Where each output is from
            a different output head
        """
        thetas = super.forward(input_val)
        if self._shared_g_theta:
            return [self._g_theta_out_layer[0](theta) for theta in thetas]
        else:
            if len(thetas) != len(self._g_theta_out_layer):
                raise Exception ("number of theta output heads must be \
                                  must be the same as num of g's (\
                                  function generators)")
            return [layer(thetas) for (layer, theta)
                    in zip(self._g_theta_out_layer, thetas)]
