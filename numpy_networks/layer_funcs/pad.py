import numpy as np

"""
Padding 4D tensors
"""


def gen_f(params, output_shape):
    a, b = params

    def f(x):
        return np.pad(x, [(0, 0), (a // 2, a - a // 2), (b // 2, b - b // 2), (0, 0)], 'constant')

    return f


def gen_df_dx(params, output_shape):
    a, b = params

    def f(inputs):
        return None, ('pad df/dx', (a, b))

    return f

gen_df_dw = None
gen_df_db = None