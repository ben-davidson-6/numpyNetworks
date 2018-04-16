import numpy as np

"""
	flattens 4d output into 2D output 
"""


def gen_f(params, output_shape):
    def f(x):
        b, h, w, c = x.shape
        return x.reshape(b, h * w * c)

    return f


def gen_df_dx(params, output_shape):
    h, w, c = params
    out = np.zeros([h * w * c, h, w, c])
    # only calculate onece so no biggy
    it = np.nditer(out, flags=['multi_index'])
    while not it.finished:
        j, s, t, u = it.multi_index
        if j == (s * c * w + t * c + u):
            out[j, s, t, u] = 1.0
        it.iternext()

    def f(x):
        return out, ('flatten df/dx', None)

    return f

gen_df_dw = None
gen_df_db = None