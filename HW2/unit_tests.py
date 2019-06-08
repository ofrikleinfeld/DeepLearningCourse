import unittest
import numpy as np
import HW2.util_functions as util_functions


class UtilsTests(unittest.TestCase):

    def gradient_checker(self, f, w):
        """ Gradient check for a function f.
        Arguments:
        f -- a function that takes a single numpy array and outputs the
             the loss and the the gradients with regards to this numpy array
        w -- the weights (numpy array) to check the gradient for
        """

        loss, grad = f(w)  # Evaluate function value at with some weights vector
        h = 1e-4  # a small value, epsilon

        # Iterate over all indexes ix in x to check the gradient.
        it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            iw = it.multi_index

            # Modifying w[iw] with h defined above to compute numerical gradients
            eps = np.zeros(w.shape)
            eps[iw] = h
            loss_plus_eps = f(w + eps)[0]
            loss_minus_eps = f(w - eps)[0]

            numeric_gradient = (loss_plus_eps - loss_minus_eps) / (2 * h)

            # Compare gradients
            gradients_diff = abs(numeric_gradient - grad[iw]) / max(1, abs(numeric_gradient), abs(grad[iw]))
            self.assertLessEqual(gradients_diff, 1e-5)

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
        res = util_functions.conv2d(x, w, bias, stride=4)
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

        np.testing.assert_allclose(util_functions.conv2d(x, w, bias), expected_res, atol=0.0001)


if __name__ == '__main__':
    unittest.main()
