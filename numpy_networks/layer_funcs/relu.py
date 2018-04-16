"""
Rectified unit
"""


def gen_f(params, output_shape):
    def f(x):
        return x * (x > 0)

    return f


def gen_df_dx(params, output_shape):
    def f(inputs):
        phantom = (inputs.ndim + len(output_shape), range(inputs.ndim), ('relu', []))
        return (inputs > 0), ('relu df/dx', None)  # phantom

    return f

gen_df_dw = None
gen_df_db = None