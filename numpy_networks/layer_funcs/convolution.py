from numpy_networks.layer_funcs import linear
from numpy_networks.utils import constants
from numpy_networks.utils.indice_help import *

"""
	Convolution function and gradient calculations
	all functions have the same parameters and output shape
"""


def gen_f(params, output_shape):
    """
        arg:
            Params
                W, filter bank
                b, bias
                window_size, filter size
                skip, convolutional stride
            output_shape, output dimensions
        output:
            4d tensor after convolution

    """
    W, b, window_size, skip = params

    def f(x):
        b2, H, W2, C = x.shape
        oh, ow = H - window_size[0] + 1, W2 - window_size[1] + 1
        after_reshape = im2row(x, (window_size, skip))
        W_as_2d = np.concatenate(np.split(np.concatenate(np.split(W, window_size[0], 1), 2).squeeze(), C, 0),
                                 1).squeeze()
        after_conv = linear.gen_f((W_as_2d, b), None)(after_reshape)
        after_reshape = conv2ten(after_conv, (b2, oh, ow, W_as_2d.shape[1]))
        return after_reshape

    return f


def gen_df_dx(params, output_shape):
    W, b, window_size, skip = params

    def f(x):
        if x.shape[1] > 20:
            delta = constants.delta_1
        else:
            delta = constants.delta_2

        M = np.einsum(delta, [0, 1, 2, 3, 4, 5], W, [6, 0, 1, 7], [2, 3, 7, 4, 5, 6])
        return M, ('conv df/dx', None)

    return f


def gen_df_dw(params, output_shape):
    W, batch, window_size, skip, = params[0], params[1], params[2], params[3]

    def f(x):
        b, h, w, I = x.shape
        oh, ow = h - window_size[0] + 1, w - window_size[1] + 1
        info = b, h, w, I, oh, ow
        return x, ('conv df/dw', info)

    return f


def gen_df_db(params, output_shape):
    def f(inputs):
        return None, ('conv df/db', None)

    return f
