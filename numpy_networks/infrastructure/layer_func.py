import numpy as np


class LayerFunc:
    """
        This holds all the gradients and functions in each layer.
        There are also some flags for avoiding redundant computation.
    """

    def __init__(self, output_shape, func, params=None, max_pool=False, trainable=False):

        self.output_shape = output_shape
        self.ndim = len(output_shape)

        # contains derivatives, and function
        # builders.
        self.gen_f = func.gen_f
        self.gen_df_dx = func.gen_df_dx
        self.gen_df_dw = func.gen_df_dw
        self.gen_df_db = func.gen_df_db

        self.params = params
        self.f = self.gen_f(params, output_shape)
        self.current_input = None

        # flags
        self.max_pool = max_pool
        self.trainable = trainable
        self.have_built = False

    def update_params(self, update_weights, update_bias):
        """Given gradient, will update the layer weights"""

        if not self.trainable:
            return

        update_weights = np.squeeze(update_weights)
        update_bias = np.squeeze(update_bias)

        # some have non-trainable parameters, in addition to the
        # weights and biases
        if len(self.params) == 2:
            self.params = (self.params[0] + update_weights, self.params[1] + update_bias)
        else:
            self.params = (self.params[0] + update_weights, self.params[1] + update_bias) + self.params[2:]

        # create updated function
        self.f = self.gen_f(self.params, self.output_shape)

    def update_f(self, ):
        self.f = self.gen_f(self.params, self.output_shape)

    def output(self, inputs):
        """Save input to the function, and output f(x)"""

        # if max pool we need to save the indices for backproping
        if self.max_pool:
            self.indices = self.f(inputs)
            out = np.take(inputs, self.indices)
        else:
            out = self.f(inputs)

        # save input
        self.current_input = inputs

        return out

    def d_dinput(self, ):
        """Get gradient of layer w.r.t. input"""
        if self.max_pool:
            df_dx = self.gen_df_dx((self.indices, self.current_input.shape),
                                   self.output_shape)(self.current_input)
        else:
            # if its trainable then the parameters will have changed and we
            # need to rebuild f
            if self.trainable:
                self.df = self.gen_df_dx(self.params, self.output_shape)
                df_dx = self.df(self.current_input)
            else:
                # if its not trainable it is fixed, so we only have to
                # make it once
                if self.have_built:
                    df_dx = self.df(self.current_input)
                else:
                    self.df = self.gen_df_dx(self.params, self.output_shape)
                    df_dx = self.df(self.current_input)
                    self.have_built = True
        return df_dx

    def d_dweight(self, ):
        """Layer gradietn w.r.t. weights"""
        return self.gen_df_dw(self.params, self.output_shape)(self.current_input)

    def d_dbias(self, ):
        """Layer gradietn w.r.t. bias"""
        out = self.gen_df_db(self.params, self.output_shape)(self.current_input)
        return out

    def del_input(self, ):
        """Delete input"""
        self.current_input = None