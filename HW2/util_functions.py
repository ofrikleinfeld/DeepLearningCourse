import numpy as np


def softmax(x):

    if len(x.shape) == 1:
        z = x - np.max(x)
        exp_z = np.exp(z)
        exp_sum = np.sum(exp_z)

    else:
        z = x - np.max(x, axis=1)[:, np.newaxis]
        exp_z = np.exp(z)
        exp_sum = np.sum(exp_z, axis=1)[:, np.newaxis]

    return exp_z / exp_sum


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return relu(x) / np.abs(x + 1e-5)


def sigmoid(x):
    # b = np.max(x, axis=0)[:, np.newaxis]
    # y = np.exp(x - b)
    # return y / np.sum(y, axis=1)[:, np.newaxis]
    z = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return z


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(input_, weights, biases):
    """ Implements a fully connected liner transformation operation.
    Will perform using matrix multiplication operations
    Arguments:
    inputs -- an input signal with the dimensions of N X d_l_minus_1
         N is the batch size, d_l_minus_1 is number of neurons in the previous layer
    weights -- the weights to perform transformation with. has a dimension of d_l x d_l_minus_1
    biases - bias terms. always 1 dimension array of d_l neurons
    """
    d_l, d_l_minus_1 = weights.shape
    weights_transposed = weights.T
    biases_shaped = biases.reshape(1, d_l)  # in order to enable broadcasting
    return input_ @ weights_transposed + biases_shaped


# def max_pool2d(input_, kernel_size=2):
#     """ Implements a 2-D max pooling operation.
#     No ability here to control the stride - it will always be identical to kernel size
#     Arguments:
#     inputs -- an input signal with the dimensions of N X C X H X W
#          N is the batch size, C is number of channels, and H and W
#          are the input height and width
#     kernel_size -- max pooling spatial area (both width and height)
#     """
#     def ceil_(x, y):
#         return int(np.ceil(x/y))
#
#     N, C, H, W = input_.shape
#
#     if H % kernel_size != 0 or W % kernel_size != 0:
#         h_out = ceil_(H, kernel_size)
#         w_out = ceil_(W, kernel_size)
#         size = (N, C, h_out * kernel_size, w_out * kernel_size)
#         input_padded = np.full(size, -np.inf)
#         input_padded[..., :H, :W] = input_
#         input_shaped = input_padded.reshape(N, C, h_out, kernel_size, w_out, kernel_size)
#
#     else:
#         h_out = H // kernel_size
#         w_out = W // kernel_size
#         input_shaped = input_.reshape(N, C, h_out, kernel_size, w_out, kernel_size)
#
#     res = input_shaped.max(axis=(3, 5))
#     return res
#
#
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv2d(input_, weights, biases, stride=1, padding=0):
    """ Implements a 2-D convolution operation.
    Will perform using matrix multiplication operations
    Arguments:
    inputs -- an input signal with the dimensions of N X C X H X W
         N is the batch size, C is number of channels, and H and W
         are the input height and width
    weights -- the kernel to perform convolution with. has a dimension of D x h X w
        where D is number of filters to apply, h and w are the kernel dimensions
    biases - bias terms. always 1 dimension array of shape D
    """

    D, C, h, w = weights.shape
    N, C, H, W = input_.shape
    h_out = (H - h + 2 * padding) / stride + 1
    w_out = (W - w + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(input_, h, w, padding=padding, stride=stride)
    W_col = weights.reshape(D, -1)

    biases = biases.reshape(D, 1)  # in order to enable broadcasting
    out = W_col @ X_col + biases
    out = out.reshape(D, h_out, w_out, N)
    out = out.transpose(3, 0, 1, 2)

    return out, X_col


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]

