import numpy as np

from numpy_networks.utils import mult_helper


class GraphLayer:
    def __init__(self, layer_func, name, ):
        self.layer_func = layer_func
        self.name = name

    def output(self, inputs):
        """output of the layer given some input"""
        return self.layer_func.output(inputs)

    def mult_tensors(self, tensor_1, tensor_2, info):
        """
            args:
                tensor_1, ND-tensor which is always the back prop
                          gradient
                tensor_2, MD-tensor which is the gradient of the
                          current layer
                info,     tuple containing the type of multiplication
                          we need to carry out, and any information we
                          may need to perform this
            output:
                \sum_{appropriate} tensor_1 tensor_2

            Based on the type of tensor_2 we get we ship it off to
            something which carries out the contraction as quick as
            I could manage. There is a naive way to do this uniformly
            but it requires building huge tensors.
        """
        form = info[0]
        if form == 'cross_entropy df/dx':
            return mult_helper.cross_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'linear df/dx':
            return mult_helper.linear_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'linear df/dw':
            return mult_helper.linear_dfdw(tensor_1, tensor_2, info[1])
        elif form == 'linear df/db':
            return mult_helper.linear_dfdb(tensor_1, tensor_2, info[1])
        elif form == 'relu df/dx':
            return mult_helper.relu_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'pad df/dx':
            return mult_helper.pad_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'conv df/dx':
            return mult_helper.conv_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'conv df/dw':
            return mult_helper.conv_dfdw(tensor_1, tensor_2, info[1])
        elif form == 'conv df/db':
            return mult_helper.conv_dfdb(tensor_1, tensor_2, info[1])
        elif form == 'max_pool df/dx':
            return mult_helper.max_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'reshape df/dx':
            return mult_helper.reshape_dfdx(tensor_1, tensor_2, info[1])
        elif form == 'flatten df/dx':
            return mult_helper.flatten_dfdx(tensor_1, tensor_2, info[1])

    def backprop(self, product, learn):
        """
            args:
                product, back prop'd gradient
                learn, learning rate

            First updates any parameters, by calculating the
            gradient dloss/dweight. Then back props the gradient.
            finally we delete the input to the layer to save memory.
        """
        if self.layer_func.trainable:
            self.update_param(product, learn)
        product = self.mult_tensors(product, *self.layer_func.d_dinput())
        self.layer_func.del_input()
        return product

    def update_param(self, product, learn):
        """
            args:
                product, backpropd gradient
                learn, learning rate

            Update the weights using NORMALISED gradients
        """
        if self.layer_func.d_dweight() is None:
            return
        weight_up = self.mult_tensors(product, *self.layer_func.d_dweight()).squeeze()
        weight_up = weight_up / (1 + np.sqrt((weight_up * weight_up).sum()))

        bias_up = self.mult_tensors(product, *self.layer_func.d_dbias()).squeeze()
        bias_up = bias_up / (1 + np.sqrt((bias_up * bias_up).sum()))
        self.layer_func.update_params(-learn * weight_up,
                                      -learn * bias_up)

    def set_params(self, params):
        """takes parameter tuple and updates layers parameters
           due to how we construct functions for each layer we
           need to update the functions at the end also
        """
        if self.layer_func.params is None:
            return
        if len(self.layer_func.params) == 2:
            self.layer_func.params = (params[0], params[1])
        else:
            self.layer_func.params = (params[0], params[1]) + self.layer_func.params[2:]
        self.layer_func.update_f()

    def get_params(self, ):
        return self.layer_func.params[:2]

    def is_trainable(self):
        return self.layer_func.trainable
