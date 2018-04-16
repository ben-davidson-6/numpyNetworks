import numpy as np

"""
Various methods to manipulate tensors into appropriate shapes
to utilise existing numpy functionality.
"""


def offset_x(s, t, batch, h, w, I, oh, ow):
    """
        This will help us with eqn (23). This will return for a given
        s,t the matrix of values X_i,j',k',l where j', and k' really
        index all j and k such that j+s, k+t are valid values of X_{i,j+s,k+t,l}

        args
            important args
                s, 	row offset
                t, 	col offset
                oh,	num row positions we have to consider
                ow, num col positions we have to consider
            boring args
                All used to know how far we need to move
                indices to go across,down, through etc
                batch,
                h,
                w,
                I,
    """

    # get first window of vals
    # imagin we have picked some window of X to convolve with
    # Then we want thos indices which are s down and t along
    a = (np.arange(oh) + s)[:, None] * w * I + (np.arange(ow) + t)[None, :] * I

    # batch and channel offsets
    batch_offset = np.arange(batch) * h * w * I
    channel_off = np.arange(I)

    # this will construct a 4D tensor of the form described previously
    a = a[None, :, :] + batch_offset[:, None, None]
    a = a[:, :, :, None] + channel_off[None, None, None, :]

    return a


####################################################################
######################rearrange#####################################
####################################################################
def im2row(x, params):
    """
        args
                x, 4D tensor: batch x height x width x channels
                params,
                    windows_size, filter size
                    skip, convolutional stride
        output
            2D tensor for convolution with the following order
                _______________________
                |	along down channel |
                |O_w                   |
                |O_h                   |
                |batch                 |
                |.                     |
                |.                     |
                |.                     |
                |______________________|

        This is inspired by code found at
        http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    """
    window_size, skip = params
    b, M, N, C = x.shape
    col_extent = N - window_size[1] + 1
    row_extent = M - window_size[0] + 1

    # get first window without other channels
    start_idx = np.arange(window_size[0])[:, None] * N * C + np.arange(window_size[1]) * C

    # Get actual first window with channel
    start_idx = start_idx.ravel()[None, :] + np.arange(C)[:, None]

    # Get the offsets within each channel
    offset_idx = np.arange(row_extent)[:, None] * N * C + np.arange(col_extent) * C

    # get all patches within a single batch
    within_window = start_idx.ravel()[None, :] + offset_idx.ravel()[:, None][::skip]

    # get batch and depth offsets
    batch_ofset = np.arange(b)[:, None, None] * M * N * C

    final_indices = np.concatenate(within_window[None, :, :] + batch_ofset,
                                   axis=0)
    return np.take(x, final_indices)


def conv2ten(x, params):
    """
        Take the output of 2D convolution and rearrange it into a
        4D tensor.
    """
    batch, oh, ow, out_channel = params
    batches = np.split(x, batch, axis=0)
    with_batch_dim = np.stack(batches, axis=0)
    ohs = np.split(with_batch_dim, oh, axis=1)
    with_all_dims = np.stack(ohs, axis=1)
    return with_all_dims
