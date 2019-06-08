import numpy as np
import HW2.util_functions as util_functions

if __name__ == '__main__':
    np.random.seed(1)
    # x = np.random.rand(4, 3, 227, 227)
    # w = np.random.rand(96, 3, 11, 11)
    # bias = np.random.rand(96)
    # util_functions.conv2d(x, w, bias)
    x = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]])
    weight = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    print(x)
    print(weight)

    stride = 1
    C = 1
    H, W = x.shape
    w, h = weight.shape

    w_out = (H - h) // stride + 1
    h_out = (W - w) // stride + 1

    i0 = np.repeat(np.arange(h), w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(h_out), w_out)

    j0 = np.tile(np.arange(w), h * C)
    j1 = stride * np.tile(np.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), h * w).reshape(-1, 1)

    cols = x[i, j]
    print(cols)
    print(cols.shape)

    rows = weight.reshape(1, C * h * w)
    print(rows.shape)
    print(rows)

    conv_res = rows @ cols
    print(conv_res.shape)
    print(conv_res.reshape(C, h_out, w_out))

