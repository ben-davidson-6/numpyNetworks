import numpy as np

"""
	reshapes a flat batch of images into a 
	b \times h \times w \times 1 tensor 
"""


def gen_f(params, output_shape):
    h, w = params

    def f(x):
        return x.reshape(-1, h, w, 1)

    return f


def gen_df_dx(params, output_shape):
    h, w = params
    out = np.zeros([h, w, h * w])
    # only calculate onece so no biggy
    for i in range(h):
        for j in range(w):
            for s in range(h * w):
                if (i * w + j) == s:
                    out[i, j, s] = 1.0

    def f(x):
        return out, ('reshape df/dx', (h, w))

    return f

gen_df_dw = None
gen_df_db = None