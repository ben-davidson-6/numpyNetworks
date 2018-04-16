import numpy as np

"""
	Perform 2 max pooling on a 4D tensor
"""


def gen_f(params, output_shape):
    skip, fh, fw = params

    def f(x):
        """
            First this will reshape the tensor into a 2D
            tensor with each row having the values of a
            max pooling window in it.

            Following this we simply take the amax over
            the column. We save the indices of the max in
            each window and output this information in a 4D
            tensor.

            Note we do NOT output the actual maxes

        """
        batch, H, W, C = x.shape

        col_extent = W - fw + 1
        row_extent = H - fh + 1

        # get first window without other channels
        start_idx = np.arange(fh)[:, None] * W * C + np.arange(fw) * C

        # Get the offsets along in each channel
        offset_idx = np.arange(col_extent) * C

        # get all along patches within a single batch
        within_window = start_idx.ravel()[None, :] + offset_idx.ravel()[:, None][::skip]

        # Get the offsets down in each channel
        offset_idx = np.arange(row_extent) * W * C

        # get all patches in single channel but need to rearrange them
        within_window = within_window[:, :, None] + offset_idx.ravel()[None, None, :][:, :, ::skip]

        # all patches in single channel
        within_window = np.concatenate(np.split(within_window, H // skip, axis=2), 0).squeeze()

        # channel offsets
        channel_off = np.arange(C)
        within_channel = within_window[:, :, None] + channel_off[None, None, :]
        within_channel = np.concatenate(np.split(within_channel, C, axis=2), 0).squeeze()

        # batch offsets
        batch_off = np.arange(batch) * W * C * H
        within_batch = within_channel[:, :, None] + batch_off[None, None, :]
        within_batch = np.concatenate(np.split(within_batch, batch, axis=2), 0).squeeze()

        # Get max index in each patch
        indices = np.argmax(np.take(x, within_batch), 1)

        # convert this to indices for x
        final = within_batch[np.arange(within_batch.shape[0]), indices]

        # rearrange back into 4D tensor
        batches = np.split(final, batch, axis=0)
        with_batch = np.stack(batches, axis=0)
        channel = np.split(with_batch, C, axis=1)
        with_batch_channel = np.stack(channel, axis=2)
        height = np.split(with_batch_channel, H // 2, axis=1)
        with_all = np.stack(height, axis=1)

        return with_all

    return f


def gen_df_dx(params, output_shape):
    indices, dims = params

    def f(x):
        """
            I honestly dont remember!
        """
        out = np.zeros(x.shape)
        np.put(out, indices.ravel(), 1)
        b, H, W, C = x.shape

        ow = W // 2
        oh = H // 2
        fh, fw = 2, 2

        # get first window
        start_idx = np.arange(fh)[:, None] * W * C + np.arange(fw) * C

        # Get the offsets along
        offset_idx = np.arange(ow) * C

        windows_along = start_idx[:, :, None] + offset_idx.ravel()[None, None, :][::2]
        windows_along = np.concatenate(np.split(windows_along, ow, axis=2), 1).squeeze()

        # Get the offsets down
        offset_idx = np.arange(oh) * C * W
        windows = windows_along[:, :, None] + offset_idx.ravel()[None, None, :][::2]
        windows = np.concatenate(np.split(windows, oh, axis=2), 0).squeeze()

        offset_idx = np.arange(oh) * C * W
        windows = windows[None, :, :] + offset_idx.ravel()[:, None, None]
        windows = np.concatenate(np.split(windows, oh, axis=0), 0).squeeze()

        offset_idx = np.arange(ow) * C
        windows = windows[:, :, :, None] + offset_idx.ravel()[None, None, None, :]

        return np.take(out, windows), ('max_pool df/dx', dims)

    return f

gen_df_dw = None
gen_df_db = None