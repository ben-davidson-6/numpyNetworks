import numpy as np

"""
	cross entropy softmax function and gradient calculations
	all functions have the same parameters and output shape
"""


def gen_f(params, output_shape):
    """no parameters or output_shape required but kept for consistency"""

    def f(inputs):
        label = inputs[0]
        z = inputs[1]
        actual_class = np.argmax(label, axis=1)
        return (-np.choose(actual_class, z.T) + np.log(np.exp(z).sum(axis=1))).sum(axis=0)

    return f


def gen_df_dx(params, output_shape):
    def f(inputs):
        label = inputs[0]
        z = inputs[1]
        actual_class = np.argmax(label, axis=1)
        v = np.exp(z)
        v = v / v.sum(axis=1)[:, None]

        def entry_func(i, r, s):
            return -(actual_class[r] == s).astype(np.float64) + v[r, s]

        return np.fromfunction(entry_func,
                               list(output_shape) + list(z.shape),
                               dtype=np.int32).astype(np.float64), ('cross_entropy df/dx', None)

    return f

gen_df_dw = None
gen_df_db = None
