import unittest
import numpy as np
import HW2.util_functions as util_functions
import HW2.modules as nn


class UtilsTests(unittest.TestCase):

    def gradient_checker(self, f, w, min_diff=1e-5):
        """ Gradient check for a function f.
        Arguments:
        f -- a function that takes a single numpy array and outputs the
             the loss and the the gradients with regards to this numpy array
        w -- the weights (numpy array) to check the gradient for
        """
        random_state = np.random.get_state()
        np.random.set_state(random_state)
        loss, grad = f(w)  # Evaluate function value at with some weights vector
        h = 1e-4  # a small value, epsilon

        # Iterate over all indexes ix in x to check the gradient.
        it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            iw = it.multi_index

            # Modifying w[iw] with h defined above to compute numerical gradients
            eps = np.zeros(w.shape)
            eps[iw] = h

            np.random.set_state(random_state)
            loss_plus_eps = f(w + eps)[0]

            np.random.set_state(random_state)
            loss_minus_eps = f(w - eps)[0]

            numeric_gradient = (loss_plus_eps - loss_minus_eps) / (2 * h)

            # Compare gradients
            gradients_diff = abs(numeric_gradient - grad[iw]) / max(1, abs(numeric_gradient), abs(grad[iw]))
            self.assertLessEqual(gradients_diff, min_diff)

            it.iternext()  # Step to next dimension

    def gradient_checker_batch_input(self, f, w, min_diff=1e-5):
        """ Gradient check for a function f.
        Arguments:
        f -- a function that takes a single numpy array and outputs the
             the loss and the the gradients with regards to this numpy array
        w -- the weights (numpy array) to check the gradient for
        """

        random_state = np.random.get_state()
        np.random.set_state(random_state)
        loss, grad = f(w)  # Evaluate function value at with some weights vector
        h = 1e-4  # a small value, epsilon

        # Iterate over all indexes ix in x to check the gradient.
        batch_size = w.shape[0]
        for i in range(batch_size):
            sample_input = w[i]
            sample_grad = grad[i]
            it = np.nditer(sample_input, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                iw = it.multi_index

                # Modifying w[iw] with h defined above to compute numerical gradients
                eps = np.zeros(sample_input.shape)
                eps[iw] = h

                np.random.set_state(random_state)
                loss_plus_eps = f(np.expand_dims(sample_input, axis=0) + eps)[0]

                np.random.set_state(random_state)
                loss_minus_eps = f(np.expand_dims(sample_input, axis=0) - eps)[0]

                numeric_gradient = (loss_plus_eps - loss_minus_eps) / (2 * h)

                # Compare gradients
                gradients_diff = abs(numeric_gradient - sample_grad[iw]) / max(1, abs(numeric_gradient), abs(sample_grad[iw]))
                self.assertLessEqual(gradients_diff, min_diff)

                it.iternext()  # Step to next dimension

    def gradient_checker_weights(self, f, w, min_diff=1e-5):
        """ Gradient check for a function f.
        Arguments:
        f -- a function that takes a single numpy array and outputs the
             the loss and the the gradients with regards to this numpy array
        w -- the weights (numpy array) to check the gradient for
        """
        random_state = np.random.get_state()
        np.random.set_state(random_state)
        loss, grad = f(w)  # Evaluate function value at with some weights vector
        h = 1e-4  # a small value, epsilon

        for i in range(len(loss)):
            # Iterate over all indexes ix in x to check the gradient.
            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                iw = it.multi_index

                # Modifying w[iw] with h defined above to compute numerical gradients
                eps = np.zeros(w.shape)
                eps[iw] = h

                np.random.set_state(random_state)
                loss_plus_eps = f(w + eps)[0]

                np.random.set_state(random_state)
                loss_minus_eps = f(w - eps)[0]

                numeric_gradient = (loss_plus_eps - loss_minus_eps) / (2 * h)

                # Compare gradients
                current_grad = grad[i][iw]
                current_numeric_grad = numeric_gradient[i]

                gradients_diff = abs(current_numeric_grad - current_grad) / max(1, abs(current_numeric_grad), abs(current_grad))
                self.assertLessEqual(gradients_diff, min_diff)

                it.iternext()  # Step to next dimension

    def test_softmax_1(self):
        input = np.array([[2, 3, 5, 1, 7], [4, 3, 0, 5, 5], [-1, 2, 0, -3, -4]])
        output = np.array([[0.00579425, 0.01575041, 0.11638064, 0.00213159, 0.85994311],
                           [0.14656828, 0.05391946, 0.00268449, 0.39841389, 0.39841389],
                           [0.04168587, 0.83728318, 0.11331396, 0.00564157, 0.00207542]])
        np.testing.assert_allclose(util_functions.softmax(input), output, atol=0.0001)

    def test_softmax_2(self):
        input = np.array([[2, 3, 5, 1, 7]])
        output = np.array([[0.00579425, 0.01575041, 0.11638064, 0.00213159, 0.85994311]])
        np.testing.assert_allclose(util_functions.softmax(input), output, atol=0.0001)

    def test_softmax_3(self):
        input = np.array([2, 3, 5, 1, 7])
        output = np.array([0.00579425, 0.01575041, 0.11638064, 0.00213159, 0.85994311])
        np.testing.assert_allclose(util_functions.softmax(input), output, atol=0.0001)

    def test_softmax_4(self):
        input = np.array([[2, 3, 5, 1, 7], [4, 3, 0, 5, 300], [-1, -6, 0, -3, -4]])
        output = np.array([[0.00579425, 0.01575041, 0.11638064, 0.00213159, 0.85994311],
                           [2.81082208e-129, 1.03404366e-129, 5.14820022e-131, 7.64060659e-129, 1.00000000e+000],
                           [0.25574518, 0.0017232, 0.69518747, 0.03461135, 0.0127328]])
        np.testing.assert_allclose(util_functions.softmax(input), output, atol=0.0001)

    def test_relu_1(self):
        input = np.array([[2, 3, 5, 1, 7], [4, 3, 0, 5, 5], [-1, 2, 0, -3, -4]])
        output = np.array([[2, 3, 5, 1, 7],
                           [4, 3, 0, 5, 5],
                           [0, 2, 0, 0, 0]])
        np.testing.assert_allclose(util_functions.relu(input), output, atol=0.0001)

    def test_relu_2(self):
        input = np.array([[-2, -3, 5, 1, 7]])
        output = np.array([[0, 0, 5, 1, 7]])
        np.testing.assert_allclose(util_functions.relu(input), output, atol=0.0001)

    def test_relu_3(self):
        input = np.array([-2, -3, 5, 1, 7])
        output = np.array([0, 0, 5, 1, 7])
        np.testing.assert_allclose(util_functions.relu(input), output, atol=0.0001)

    def test_sigmoid_1(self):
        input = np.array([[2, 3, 5, 1, 7], [-4, -3, 0, -5, 1], [-1, 2, 0, -3, -4]])
        output = np.array([[0.88079708, 0.95257413, 0.99330715, 0.73105858, 0.99908895],
                           [0.01798621, 0.04742587, 0.5, 0.00669285, 0.73105858],
                           [0.26894142, 0.88079708, 0.5, 0.04742587, 0.01798621]])
        np.testing.assert_allclose(util_functions.sigmoid(input), output, atol=0.0001)

    def test_sigmoid_2(self):
        input = np.array([[-2, -3, 5, 1, 7]])
        output = np.array([[0.11920292, 0.04742587, 0.99330715, 0.73105858, 0.99908895]])
        np.testing.assert_allclose(util_functions.sigmoid(input), output, atol=0.0001)

    def test_sigmoid_3(self):
        input = np.array([-2, -3, 5, 1, 7])
        output = np.array([0.11920292, 0.04742587, 0.99330715, 0.73105858, 0.99908895])
        np.testing.assert_allclose(util_functions.sigmoid(input), output, atol=0.0001)

    def tests_gradient_exponent_function(self):

        def exp_sum(x):
            return np.sum(np.exp(x)), np.exp(x)

        self.gradient_checker(exp_sum, np.array(102.82))  # scalar test
        self.gradient_checker(exp_sum, np.array([0, 0, 0, 0]))  # 1-D test
        self.gradient_checker(exp_sum, np.array([[4, 5], [1, 1], [0, 0]]))  # 2-D test

    def test_gradients_quad_function(self):

        def quad(x):
            return np.sum(x ** 2), x * 2

        self.gradient_checker(quad, np.array(102.82))  # scalar test
        self.gradient_checker(quad, np.array([0, 0, 0, 0]))  # 1-D test
        self.gradient_checker(quad, np.array([[4, 5], [1, 1], [0, 0]]))  # 2-D test

    def test_gradient_log_function(self):

        def log_function(x):
            return np.sum(np.log(x)), 1 / x

        self.gradient_checker(log_function, np.array(102.82))  # scalar test
        self.gradient_checker(log_function, np.array([1, 2, 33, 15]))  # 1-D test
        self.gradient_checker(log_function, np.array([[4, 5], [1, 1], [3.24, 1.1]]))  # 2-D test

    def test_gradient_relu(self):

        def relu_function(x):
            return np.sum(util_functions.relu(x)), util_functions.relu_derivative(x)

        self.gradient_checker(relu_function, np.array(102.82))  # scalar test
        self.gradient_checker(relu_function, np.array([-2, 1, 1, 1]))  # 1-D test
        self.gradient_checker(relu_function, np.array([[4, 5], [1, 1], [-5, 21]]))  # 2-D test

    def test_gradient_sigmoid(self):

        def sigmoid_function(x):
            return np.sum(util_functions.sigmoid(x)), util_functions.sigmoid_derivative(x)

        self.gradient_checker(sigmoid_function, np.array(102.82))  # scalar test
        self.gradient_checker(sigmoid_function, np.array([0, 0, 0, 0]))  # 1-D test
        self.gradient_checker(sigmoid_function, np.array([[4, 5], [1, 1], [-5, 21]]))  # 2-D test

    def test_conv2d_shape(self):
        x = np.random.rand(4, 3, 227, 227)
        w = np.random.rand(96, 3, 11, 11)
        bias = np.random.rand(96)
        res, _ = util_functions.conv2d(x, w, bias, stride=4)
        self.assertEqual(res.shape, (4, 96, 55, 55))

    def test_conv2d_res_1(self):
        # 1 sample, 1 color map, 5x5
        x = np.array([[[[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]]])

        # 1 filter of size 3x3 (for 1 color map)
        w = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])

        bias = np.zeros(1)  # 1 filter so 1 bias term

        # result of dimension (1, 1, 3, 3)
        expected_res = np.array([[[[-13., -20., -17.],
                                [-18., -24., -18.],
                                [13.,  20.,  17.]]]])

        z, _ = util_functions.conv2d(x, w, bias)

        np.testing.assert_allclose(z, expected_res, atol=0.0001)

    def test_linear_1(self):
        x = np.array([[1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[2., 70.]])

        np.testing.assert_allclose(util_functions.linear(x, w, b), expected_res, atol=0.0001)

    def test_linear_2(self):
        x = np.array([[1, 2, 33, 15], [1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[2., 70.], [2., 70.]])

        np.testing.assert_allclose(util_functions.linear(x, w, b), expected_res, atol=0.0001)

    def test_linear_3(self):
        x = np.array([[1, 2, 0, -1], [1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.array([-2., -1.])
        expected_res = np.array([[0., 3.], [0., 69.]])

        np.testing.assert_allclose(util_functions.linear(x, w, b), expected_res, atol=0.0001)

    def test_linear_with_relu_1(self):
        x = np.array([[1, 2, 33, 15], [1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[2., 70.], [2., 70.]])

        np.testing.assert_allclose(util_functions.relu(util_functions.linear(x, w, b)), expected_res, atol=0.0001)

    def test_linear_with_relu_2(self):
        x = np.array([[1, 2, 0, -1], [1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.array([-5., -1.])
        expected_res = np.array([[0., 3.], [0., 69.]])

        np.testing.assert_allclose(util_functions.relu(util_functions.linear(x, w, b)), expected_res, atol=0.0001)

    def test_linear_module_1(self):
        x = np.array([[1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[2., 70.]])

        linear_layer = nn.Linear(4, 2)
        linear_layer.set_weights(w)
        linear_layer.set_biases(b)

        z = linear_layer(x)

        np.testing.assert_allclose(z, expected_res, atol=0.0001)

    def test_linear_module_relu_1(self):
        x = np.array([[1, 2, 0, -1], [1, 2, 33, 15]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.array([-5., -1.])
        expected_res = np.array([[0., 3.], [0., 69.]])

        linear_layer = nn.Linear(4, 2)
        linear_layer.set_weights(w)
        linear_layer.set_biases(b)

        relu_layer = nn.Relu()
        z = linear_layer(x)
        a = relu_layer(z)

        np.testing.assert_allclose(a, expected_res, atol=0.0001)

    def test_linear_module_relu_2(self):
        x = np.array([[1, 2, 0, -1], [1, 2, -1, -2]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.array([-5., -1.])
        expected_res = np.array([[0., 3.], [0., 1.]])

        linear_layer = nn.Linear(4, 2)
        linear_layer.set_weights(w)
        linear_layer.set_biases(b)

        relu_layer = nn.Relu()
        z = linear_layer(x)
        a = relu_layer(z)

        np.testing.assert_allclose(a, expected_res, atol=0.0001)

    def test_linear_module_softmax_1(self):
        x = np.array([[1, 2, 0, -1], [1, 2, -1, -2]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[.119202, .88079], [.5, .5]])

        linear = nn.Linear(4, 2)
        softmax = nn.Softmax()
        linear.set_weights(w)
        linear.set_biases(b)

        z = linear(x)
        a = softmax(z)

        np.testing.assert_allclose(a, expected_res, atol=0.0001)

    def test_gradient_softmax_layer_batch(self):

        def softmax_layer(x):
            softmax = nn.Softmax()
            num_classes = x.shape
            label = np.zeros(num_classes)
            label[:, 0] = 1
            dist = softmax(x)
            loss = -np.log(np.sum(dist * label, axis=1))
            grad = softmax.backward(label)
            return loss, grad

        self.gradient_checker_batch_input(softmax_layer, np.array([[1, 2, -1, -2], [1, 2, -1, -2]]))  # batch size test

    def test_gradient_relu_layer_batch(self):

        def relu_layer(x):
            relu = nn.Relu()
            softmax = nn.Softmax()

            a = softmax(relu(x))

            num_classes = x.shape
            labels = np.zeros(num_classes)
            labels[:, 0] = 1

            loss = -np.log(np.sum(a * labels, axis=1))
            softmax_grad = softmax.backward(labels)
            relu_grad = relu.backward(softmax_grad)
            return loss, relu_grad

        self.gradient_checker_batch_input(relu_layer, np.array([[1, 2, -1, -2], [1, 2, -1, -2], [1, 2, -1, -2]]))  # batch size test

    def test_gradient_sigmoid_layer_batch(self):

        def sigmoid_layer(x):
            sigmoid = nn.Sigmoid()
            softmax = nn.Softmax()

            a = softmax(sigmoid(x))

            num_classes = x.shape
            labels = np.zeros(num_classes)
            labels[:, 0] = 1

            loss = -np.log(np.sum(a * labels, axis=1))
            softmax_grad = softmax.backward(labels)
            sigmoid_grad = sigmoid.backward(softmax_grad)

            return loss, sigmoid_grad

        self.gradient_checker_batch_input(sigmoid_layer, np.array([[1, 2, -1, -2], [1, 2, -1, -2]]))  # batch size test

    def test_gradient_linear_layer_batch(self):

        def linear_layer(z):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification
            relu_layer = nn.Relu()
            linear = nn.Linear(in_dimension=2, out_dimension=5)
            softmax = nn.Softmax()

            a_L_mins_1 = relu_layer(z)
            z_L = linear(a_L_mins_1)
            a_L = softmax(z_L)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            layer_L_grad = linear.backward(softmax_grad)
            relu_grad = relu_layer.backward(layer_L_grad)

            return loss, relu_grad

        self.gradient_checker_batch_input(linear_layer, np.array([[22, 3], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]))  # batch size test
        self.gradient_checker_batch_input(linear_layer, np.array([[1, 3], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]))

    def test_gradient_flatten_layer_batch(self):

        # z size is 2x3x4x4
        z = np.array([[[[1, 2, 0, 1], [0, 0, 2, 1], [2, 1, 2, 1], [2, 0, 1, 2]],
                       [[2, 1, 2, 1], [2, 2, 0, 0], [2, 0, 2, 0], [0, 1, 2, 1]],
                       [[0, 2, 1, 2], [2, 1, 0, 0], [1, 1, 0, 1], [2, 1, 2, 2]]],
                      [[[2, 1, 0, 0], [2, 2, 0, 1], [0, 0, 1, 0], [0, 2, 1, 1]],
                       [[1, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 1], [1, 2, 1, 1]],
                       [[2, 2, 1, 0], [0, 2, 0, 1], [0, 0, 2, 2], [0, 1, 2, 0]]]])

        def flatten(x):

            flatten_ = nn.Flatten()
            linear = nn.Linear(in_dimension=48, out_dimension=4)
            softmax = nn.Softmax()

            # forward
            flatten_x = flatten_(x)
            dist = softmax(linear(flatten_x))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten_.backward(linear_grad)
            return loss, flatten_grad

        self.gradient_checker_batch_input(flatten, z)

    def test_conv2d_res_shape_padding(self):
        # 1 sample, 1 color map, 10x10
        x = np.array([[[
            [0, 0, 0, 0, 0, -1, 2, 1, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 1, -1, 0],
            [0, 1, 2, 3, 0, -1, 2, -1, -2, 1],
            [0, 4, 5, 6, 0, 0, 0, 1, 2, 0],
            [0, 4, 5, 6, 0, 0, 0, 1, 2, 0],
            [0, 4, 5, 6, 0, 0, 0, 1, 2, 0],
            [0, 7, 8, 9, 0, 1, 2, -1, 0, 1],
            [0, 7, 8, 9, 0, 1, 2, -1, 0, 1],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        ]]])

        # 1 filter of size 3x3 (for 1 color map)
        w = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])

        bias = np.zeros(1)  # 1 filter so 1 bias term

        # result of dimension (1, 1, 10, 10)
        res, _ = util_functions.conv2d(x, w, bias, stride=1, padding=1)
        expected_shape = np.array((1, 1, 10, 10))

        np.testing.assert_allclose(res.shape, expected_shape, atol=0.0001)

    def test_gradient_conv_layer_batch(self):

        x_ = np.array([  # shape of 6x1x2x2
            [[[1, 3], [0, 1]]],
            [[[1, 2], [0, 1]]],
            [[[1, 2], [0, 2]]],
            [[[1, 2], [2, -1]]],
            [[[1, 2], [0, 1]]],
            [[[1, 2], [-1, -1]]]
        ])

        def conv_layer(x):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1)
            relu = nn.Relu()
            flatten = nn.Flatten()
            linear = nn.Linear(16, 4)
            softmax = nn.Softmax()

            # forward pass
            a = relu(conv(x))
            a_flatten = flatten(a)
            dist = softmax(linear(a_flatten))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            # backward pass
            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten.backward(linear_grad)
            relu_grad = relu.backward(flatten_grad)
            conv_grad = conv.backward(relu_grad)

            return loss, conv_grad

        self.gradient_checker_batch_input(conv_layer, x_, min_diff=1e-3)  # batch size test

    def test_gradient_2_conv_layers_batch(self):

        x_ = np.array([  # shape 6x1x3x3
        [[[-1,  1,  1],
         [ 1,  1,  0],
         [-1,  0,  1]]],
       [[[-1,  0,  1],
         [ 0, -1,  0],
         [ 1,  1, -1]]],
       [[[ 1,  1, -1],
         [ 0,  1, -1],
         [-1, -1,  1]]],
       [[[ 1,  1,  1],
         [-1, -1,  0],
         [ 0,  1,  1]]],
       [[[-1,  0, -1],
         [ 0,  1, -1],
         [ 1,  1,  0]]],
       [[[ 0,  0,  0],
         [-1, -1,  1],
         [ 0, -1, -1]]]
        ])

        def conv_layer(x):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
            relu1 = nn.Relu()
            conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2)
            relu2 = nn.Relu()
            flatten = nn.Flatten()
            linear = nn.Linear(4, 2)
            softmax = nn.Softmax()

            # forward pass
            a = relu1(conv1(x))
            a = relu2(conv2(a))
            a_flatten = flatten(a)
            dist = softmax(linear(a_flatten))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten.backward(linear_grad)
            relu2_grad = relu2.backward(flatten_grad)
            conv2_grad = conv2.backward(relu2_grad)
            relu1_grad = relu1.backward(conv2_grad)
            conv1_grad = conv1.backward(relu1_grad)

            return loss, conv1_grad

        self.gradient_checker_batch_input(conv_layer, x_, min_diff=1e-3)

    def test_gradient_conv_layers_with_sigmoid(self):

        x_ = np.array([  # shape 6x1x3x3
            [[[-1, 1, 1],
              [1, 1, 0],
              [-1, 0, 1]]],
            [[[-1, 0, 1],
              [0, -1, 0],
              [1, 1, -1]]],
            [[[1, 1, -1],
              [0, 1, -1],
              [-1, -1, 1]]],
            [[[1, 1, 1],
              [-1, -1, 0],
              [0, 1, 1]]],
            [[[-1, 0, -1],
              [0, 1, -1],
              [1, 1, 0]]],
            [[[0, 0, 0],
              [-1, -1, 1],
              [0, -1, -1]]]
        ])

        def conv_layer(x):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
            sigmoid1 = nn.Sigmoid()
            conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2)
            sigmoid2 = nn.Sigmoid()
            flatten = nn.Flatten()
            linear = nn.Linear(4, 2)
            softmax = nn.Softmax()

            # forward pass
            a = sigmoid1(conv1(x))
            a = sigmoid2(conv2(a))
            a_flatten = flatten(a)
            dist = softmax(linear(a_flatten))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten.backward(linear_grad)
            sigmoid2_grad = sigmoid2.backward(flatten_grad)
            conv2_grad = conv2.backward(sigmoid2_grad)
            sigmoid1_grad = sigmoid1.backward(conv2_grad)
            conv1_grad = conv1.backward(sigmoid1_grad)

            return loss, conv1_grad

        self.gradient_checker_batch_input(conv_layer, x_, min_diff=1e-3)

    def test_gradient_linear_wrt_weights(self):

        z = np.array([[1, 3, 0], [1, 2, -1], [1, 2, 4], [1, 2, -3], [1, 2, -2], [1, 2, 1]])
        w_ = np.array([[2, 0, 2], [1, 2, 2]])

        def linear_layer(w):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification
            linear = nn.Linear(in_dimension=3, out_dimension=2)
            linear.set_weights(w)
            softmax = nn.Softmax()

            # forward
            dist = softmax(linear(z))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            w_grad = linear.w_grad

            return loss, w_grad

        self.gradient_checker_weights(linear_layer, w_)

    def test_gradient_linear_wrt_biases(self):

        z = np.array([[1, 3, 0], [1, 2, -1], [1, 2, 4], [1, 2, -3], [1, 2, -2], [1, 2, 1]])
        b_ = np.array([2, 0])

        def linear_layer(b):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification
            linear = nn.Linear(in_dimension=3, out_dimension=2)
            linear.set_biases(b)
            softmax = nn.Softmax()

            # forward
            dist = softmax(linear(z))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            b_grad = linear.b_grad

            return loss, b_grad

        self.gradient_checker_weights(linear_layer, b_)

    def test_gradient_conv_wrt_biases(self):

        x = np.array([  # shape 6x1x3x3
            [[[-1, 1, 1],
              [1, 1, 0],
              [-1, 0, 1]]],
            [[[-1, 0, 1],
              [0, -1, 0],
              [1, 1, -1]]],
            [[[1, 1, -1],
              [0, 1, -1],
              [-1, -1, 1]]],
            [[[1, 1, 1],
              [-1, -1, 0],
              [0, 1, 1]]],
            [[[-1, 0, -1],
              [0, 1, -1],
              [1, 1, 0]]],
            [[[0, 0, 0],
              [-1, -1, 1],
              [0, -1, -1]]]
        ])

        b_ = np.array([0.5, 2, 0])

        def conv(b):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification
            conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2)
            relu = nn.Relu()
            flatten = nn.Flatten()
            linear = nn.Linear(in_dimension=12, out_dimension=4)
            softmax = nn.Softmax()

            conv.set_biases(b)

            # forward
            a = flatten(relu(conv(x)))
            dist = softmax(linear(a))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten.backward(linear_grad)
            relu_grad = relu.backward(flatten_grad)
            conv_grad = conv.backward(relu_grad)

            b_grad = conv.b_grad

            return loss, b_grad

        self.gradient_checker_weights(conv, b_, min_diff=1e-3)

    def test_gradient_2_conv_layers_batch(self):

        x_ = np.array([  # shape 6x1x3x3
        [[[-1,  1,  1],
         [ 1,  1,  0],
         [-1,  0,  1]]],
       [[[-1,  0,  1],
         [ 0, -1,  0],
         [ 1,  1, -1]]],
       [[[ 1,  1, -1],
         [ 0,  1, -1],
         [-1, -1,  1]]],
       [[[ 1,  1,  1],
         [-1, -1,  0],
         [ 0,  1,  1]]],
       [[[-1,  0, -1],
         [ 0,  1, -1],
         [ 1,  1,  0]]],
       [[[ 0,  0,  0],
         [-1, -1,  1],
         [ 0, -1, -1]]]
        ])

        def conv_layer(x):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
            relu1 = nn.Relu()
            conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2)
            relu2 = nn.Relu()
            flatten = nn.Flatten()
            linear = nn.Linear(4, 2)
            softmax = nn.Softmax()

            # forward pass
            a = relu1(conv1(x))
            a = relu2(conv2(a))
            a_flatten = flatten(a)
            dist = softmax(linear(a_flatten))

            # backward
            labels = np.zeros(dist.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(dist * labels, axis=1))

            softmax_grad = softmax.backward(labels)
            linear_grad = linear.backward(softmax_grad)
            flatten_grad = flatten.backward(linear_grad)
            relu2_grad = relu2.backward(flatten_grad)
            conv2_grad = conv2.backward(relu2_grad)
            relu1_grad = relu1.backward(conv2_grad)
            conv1_grad = conv1.backward(relu1_grad)

            return loss, conv1_grad

        self.gradient_checker_batch_input(conv_layer, x_, min_diff=1e-3)

    # def test_max_pooling(self):
    #     x = np.array([[[[ 1, -1],
    #      [-1,  2]]],
    #    [[[ 1,  0],
    #      [ 3,  1]]],
    #    [[[-1,  3],
    #      [ 1,  0]]]])
    #
    #     expected_res = np.array([[[[2]]], [[[3]]], [[[3]]]])
    #     expected_idxs = np.array([3, 2, 1])
    #     a, max_idx, _ = util_functions.max_pool2d(x, kernel_size=2, stride=1)
    #     np.testing.assert_allclose(a, expected_res, atol=0.0001)
    #     np.testing.assert_allclose(max_idx, expected_idxs, atol=1e-4)
    #
    # def test_gradient_maxpool_layer_batch(self):
    #
    #     # x_ size is 2x3x4x4
    #     x_ = np.array(
    #     [[[[ 0,  0,  3,  3],
    #      [ 3,  2,  1,  1],
    #      [1,  2, -1,  3],
    #      [ 0,  2,  1, -1]],
    #
    #     [[ 2,  1,  3, -1],
    #      [ 0,  2,  0,  3],
    #      [ 0,  0,  1,  0],
    #      [ 2, -1,  1,  1]],
    #
    #     [[ 3,  1, -1, -1],
    #      [ 0, -1,  1,  0],
    #      [ 2,  1,  0, -1],
    #      [ 0,  1,  2,  3]]],
    #
    #
    #    [[[ 0,  3, -1,  2],
    #      [0,  0,  0,  2],
    #      [-1,  0,  2,  2],
    #      [ 2,  3,  3,  2]],
    #
    #     [[ 0,  2,  3,  3],
    #      [ 3, -1,  0,  1],
    #      [ 2,  2,  2,  3],
    #      [ 1,  0,  2,  2]],
    #
    #     [[-1,  3, -1,  2],
    #      [ 0, -1,  2,  3],
    #      [ 0,  0,  0, -1],
    #      [-1,  0,  0,  2]]]])
    #
    #     def max_pool(x):
    #
    #         pool = nn.MaxPool2d(kernel_size=2)
    #         flatten = nn.Flatten()
    #         linear = nn.Linear(in_dimension=12, out_dimension=4)
    #         softmax = nn.Softmax()
    #
    #         # forward
    #         a = pool(x)
    #         a_flatten = flatten(a)
    #         dist = softmax(linear(a_flatten))
    #
    #         # backward
    #         labels = np.zeros(dist.shape)
    #         labels[:, 1] = 1
    #         loss = -np.log(np.sum(dist * labels, axis=1))
    #
    #         softmax_grad = softmax.backward(labels)
    #         linear_grad = linear.backward(softmax_grad)
    #         flatten_grad = flatten.backward(linear_grad)
    #         pool_grad = pool.backward(flatten_grad)
    #         return loss, pool_grad
    #
    #     self.gradient_checker_batch_input(max_pool, x_, 1e-3)


if __name__ == '__main__':
    unittest.main()
