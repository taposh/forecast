import torch.nn as nn
import torch.nn.functional as f


class NBeatsBlock(nn.Module):
    """ Basic Building Block of N-Beats.

    This class constructs the basic building block of a 
    N-beats NN, the architecture looks like the following:

                               |-theta_forwards
        input - hidden_layers -|
                               |-theta_backwards
    Where each layer is fully connected

    Args:
        -layer_dim(int): dimension of input and outputs of
            input and hidden layers
        -thetas_dim(list/tuple): list or iterable of output
            dimensions of the theta output layers
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
                 layer_dim=1,
                 thetas_dim=None,
                 num_hidden_layers=2,
                 layer_nonlinearity=nn.ReLU,
                 layer_w_init=nn.init.xavier_uniform_,
                 layer_b_init=nn.init.zeros_):
        super().__init__()
        self._layer_dim = layer_dim
        self._num_hidden_layers = num_hidden_layers
        self._theta_heads = 2
        self._thetas_dim = (thetas_dim if thetas_dim is not
                           None else [self._layer_dim] * self._theta_heads)
        if (not (type(self._thetas_dim) == tuple
                 or type(self._thetas_dim) == list)):
            raise Exception("thetas dim must be of type list or tuple")
        assert(len(self._thetas_dim) == self._theta_heads)
        self._layers = nn.ModuleList()
        self._thetas_output_layers = nn.ModuleList()

        # input layer
        input_layer = nn.Sequential()
        linear_layer = nn.Linear(self._layer_dim, self._layer_dim)
        layer_w_init(linear_layer.weight)
        layer_b_init(linear_layer.bias)
        input_layer.add_module('block_input', linear_layer)
        if layer_nonlinearity:
            input_layer.add_module('non_linearity', layer_nonlinearity())
        self._layers.append(input_layer)

        # hidden layers
        for i in range(self._num_hidden_layers):
            hidden_layer = nn.Sequential()
            linear_layer = nn.Linear(self._layer_dim, self._layer_dim)
            layer_w_init(linear_layer.weight)
            layer_b_init(linear_layer.bias)
            hidden_layer.add_module('block_hidden_' + str(i), linear_layer)
            if layer_nonlinearity:
                hidden_layer.add_module('non_linearity', layer_nonlinearity())
            self._layers.append(hidden_layer)

        # multi-headed output
        for i in range(self._theta_heads):
            output_head = nn.Sequential()
            linear_layer = nn.Linear(self._layer_dim, self._thetas_dim[i])
            layer_w_init(linear_layer.weight)
            layer_b_init(linear_layer.bias)
            output_head.add_module('block_output_' + str(i), linear_layer)
            if layer_nonlinearity:
                output_head.add_module('non_linearity', layer_nonlinearity())
            self._thetas_output_layers.append(output_head)

    def forward(self, input_val):
        """ Feed Forward function for Block module.

        Args:
            input val(torch.tensor): input for block
        Returns:
            List of torch.tensors that are outputs of the network,
            Where each output is from a different output head
        """
        x = input_val
        for layer in self._layers:
            x = layer(x)
        if len(self._thetas_output_layers) == 1:
            return self._thetas_output_layers[0](x)
        else:
            return [layer(x) for layer in self._thetas_output_layers]
