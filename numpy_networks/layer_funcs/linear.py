import numpy as np

"""
	Function and gradients of RIGHT matrix multiplication
	XW
"""


def gen_f(params, output_shape):
    def f(x):
        return np.dot(x, params[0]) + params[1]

    return f


def gen_df_dx(params, output_shape):
    def f(inputs):
        return params[0].T, ('linear df/dx', None)

    return f


def gen_df_dw(params, output_shape):
    def f(inputs):
        return inputs, ('linear df/dw', None)

    return f


def gen_df_db(params, output_shape):
    def f(inputs):
        return np.ones([output_shape[0]]), ('linear df/db', None)

    return f
