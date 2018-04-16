import numpy as np

from numpy_networks.utils import indice_help

"""
	This is only ever called from mult_tensors.
	Each method performs something EQUIVALENT to
	given tensor contractions. 
"""


def conv_dfdw(A, B, info):
    batch, h, w, I, oh, ow = info
    row = [[], [], []]
    for s in range(3):
        for t in range(3):
            new_B = np.take(B, indice_help.offset_x(s, t, batch, h, w, I, oh, ow))
            # result is some dim and then r,u
            row[s].append(np.einsum(A, list(range(A.ndim)), new_B, [1, 2, 3, 5], [0, 5, 4]))
    stacked = []
    for r in row:
        stacked.append(np.stack(r, 2))
    return np.stack(stacked, 2)


def conv_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = A_dim[-3:] + [A_dim[-1] + 1, A_dim[-1] + 2, A_dim[-1] + 3]
    final_dim = A_dim[:-3] + [A_dim[-1] + 1, A_dim[-1] + 2, A_dim[-1] + 3]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def conv_dfdb(A, B, info):
    A_dim = list(range(A.ndim))
    return np.einsum(A, A_dim, [0, A_dim[-1]])


def linear_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = [A_dim[-2] + 1, A_dim[-2] + 3]
    final_dims = A_dim[:-2] + [A_dim[-2], A_dim[-2] + 3]
    return np.einsum(A, A_dim, B, B_dim, final_dims)


def linear_dfdw(A, B, info):
    A_dim = list(list(range(A.ndim)))
    B_dim = [A_dim[-2], A_dim[-2] + 2]
    final_dim = A_dim[:-2] + [A_dim[-2] + 2, A_dim[-2] + 1]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def linear_dfdb(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = [A_dim[-2]]
    final_dim = A_dim[:-2] + [A_dim[-2] + 1]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def relu_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    input_shape = B.ndim
    B_dim = range(A_dim[-input_shape], A_dim[-input_shape] + input_shape)
    final_dim = A_dim
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def cross_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    input_shape = B.ndim
    B_dim = range(A_dim[-input_shape], A_dim[-input_shape] + 2)
    final_dim = A_dim[:-input_shape] + B_dim[1:]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def pad_dfdx(A, B, info):
    a, b = info
    return A[:, :, a // 2:-a // 2, b // 2:-b // 2, :]


def max_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = [A_dim[-3], 5, 6, A_dim[-2]]
    final_dim = A_dim[:-4] + [A_dim[-4], 5, 6, A_dim[-1]]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def reshape_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = [A_dim[-4] + 1, A_dim[-4] + 2, A_dim[-4] + 4]
    final_dim = A_dim[:-4] + [A_dim[-4], A_dim[-1]]
    return np.einsum(A, A_dim, B, B_dim, final_dim)


def flatten_dfdx(A, B, info):
    A_dim = list(range(A.ndim))
    B_dim = [A_dim[-2] + 1, A_dim[-2] + 3, A_dim[-2] + 4, A_dim[-2] + 5]
    final_dim = A_dim[:-2] + [A_dim[-2], A_dim[-2] + 3, A_dim[-2] + 4, A_dim[-2] + 5]
    return np.einsum(A, A_dim, B, B_dim, final_dim)
