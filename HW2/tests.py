import unittest
import numpy as np
import HW2.util_functions as util_functions
import HW2.modules as nn


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

    def gradient_checker_batch_input(self, f, w):
        """ Gradient check for a function f.
        Arguments:
        f -- a function that takes a single numpy array and outputs the
             the loss and the the gradients with regards to this numpy array
        w -- the weights (numpy array) to check the gradient for
        """

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

                loss_plus_eps = f(np.expand_dims(sample_input, axis=0) + eps)[0]
                loss_minus_eps = f(np.expand_dims(sample_input, axis=0) - eps)[0]

                numeric_gradient = (loss_plus_eps - loss_minus_eps) / (2 * h)

                # Compare gradients
                gradients_diff = abs(numeric_gradient - sample_grad[iw]) / max(1, abs(numeric_gradient), abs(sample_grad[iw]))
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

    def test_liner_module_relu_1(self):
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

    def test_liner_module_relu_2(self):
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

    def test_liner_module_softmax_1(self):
        x = np.array([[1, 2, 0, -1], [1, 2, -1, -2]])
        w = np.array([[0., 1., 0., 0.], [0., 2., 2., 0.]])
        b = np.zeros(2)
        expected_res = np.array([[.119202, .88079], [.5, .5]])

        linear_with_softmax = nn.LinearWithSoftmax(4, 2)
        linear_with_softmax.set_weights(w)
        linear_with_softmax.set_biases(b)

        a = linear_with_softmax(x)

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

    def test_gradient_relu_layer(self):

        def relu_layer(x):
            relu = nn.Relu()
            loss = np.sum(relu(x))
            grad = relu.backward()
            return loss, grad

        self.gradient_checker(relu_layer, np.array([1, 2, -1, -2]))  # 1-D test

    def test_gradient_relu_layer_batch(self):

        def relu_layer(x):
            relu = nn.Relu()
            loss = np.sum(relu(x), axis=1)
            grad = relu.backward()
            return loss, grad

        self.gradient_checker_batch_input(relu_layer, np.array([[1, 2, -1, -2], [1, 2, -1, -2], [1, 2, -1, -2]]))  # batch size test

    def test_gradient_sigmoid_layer_1(self):

        def sigmoid_layer(x):
            sigmoid = nn.Sigmoid()
            loss = np.sum(sigmoid(x))
            grad = sigmoid.backward()
            return loss, grad

        self.gradient_checker(sigmoid_layer, np.array([1, 2, -1, -2]))  # 1-D test
        self.gradient_checker(sigmoid_layer, np.array([-1, 1, -1, -2]))  # another 1-D test

    def test_gradient_sigmoid_layer_batch(self):

        def sigmoid_layer(x):
            sigmoid = nn.Sigmoid()
            loss = np.sum(sigmoid(x))
            grad = sigmoid.backward()
            return loss, grad

        self.gradient_checker_batch_input(sigmoid_layer, np.array([[1, 2, -1, -2], [1, 2, -1, -2]]))  # batch size test

    def test_gradient_linear_layer_batch(self):

        def linear_layer(z):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification
            relu_layer = nn.Relu()
            a_L_mins_1 = relu_layer(z)

            linear_L = nn.LinearWithSoftmax(in_dimension=2, out_dimension=5)
            linear_L.set_weights(np.array([
                [0, 1],
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 1],

            ]))
            linear_L.set_biases(np.array([0, 0, 0, 0, 0]))

            a_L = linear_L(a_L_mins_1)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))
            layer_L_grad = linear_L.backward(labels)
            grad = layer_L_grad @ linear_L.get_weights() * relu_layer.backward()
            return loss, grad

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

        w_L = np.array([[0.6881892 , 0.52568451, 0.36596662, 0.3826301 , 0.84299256,
        0.55766545, 0.79826597, 0.59906424, 0.45119649, 0.29873219,
        0.95950442, 0.90259326, 0.71561526, 0.87848583, 0.51235738,
        0.44085556, 0.99806538, 0.17446175, 0.73219746, 0.76321222,
        0.3269503 , 0.67003723, 0.08105612, 0.17789013, 0.87016942,
        0.73115949, 0.57178014, 0.23227211, 0.37972477, 0.28508733,
        0.47240048, 0.1453711 , 0.18006982, 0.74171713, 0.00892932,
        0.521951  , 0.21183233, 0.72908835, 0.03955933, 0.21198875,
        0.02974047, 0.84043704, 0.91020544, 0.10463805, 0.96402542,
        0.26657336, 0.60137633, 0.97881282],
       [0.0192244 , 0.8428359 , 0.65203514, 0.40715368, 0.53914569,
        0.27307317, 0.54743349, 0.31933828, 0.19416524, 0.52484577,
        0.45332771, 0.23135872, 0.81965197, 0.41534829, 0.62229574,
        0.9318435 , 0.15060489, 0.49775094, 0.27773329, 0.79340574,
        0.7876512 , 0.06391876, 0.7447968 , 0.05007631, 0.30948458,
        0.81222198, 0.81970602, 0.14809552, 0.67984271, 0.46589285,
        0.91332432, 0.87325994, 0.3965656 , 0.35513036, 0.22743372,
        0.36555223, 0.03041562, 0.12730802, 0.38488217, 0.24221227,
        0.46506702, 0.7414589 , 0.70489737, 0.97213108, 0.46904931,
        0.26802368, 0.76824031, 0.78498747],
       [0.21973077, 0.2841096 , 0.42907162, 0.01979624, 0.24282574,
        0.74082669, 0.7233429 , 0.41230427, 0.35046596, 0.35530596,
        0.14382612, 0.75009273, 0.34291776, 0.86810133, 0.1739645 ,
        0.16099521, 0.84131839, 0.32737914, 0.27426773, 0.60243704,
        0.50239915, 0.91143134, 0.03127544, 0.76445815, 0.7524862 ,
        0.74814287, 0.59896241, 0.64148864, 0.05095864, 0.54036924,
        0.30603012, 0.18642942, 0.10908812, 0.7860116 , 0.79188573,
        0.99836632, 0.79999588, 0.4902652 , 0.88223844, 0.66593393,
        0.35426412, 0.54944994, 0.01750188, 0.9771826 , 0.9195825 ,
        0.03914166, 0.44874936, 0.65759457],
       [0.05798657, 0.0130046 , 0.96216221, 0.44154329, 0.80212338,
        0.7119418 , 0.48039107, 0.35212055, 0.05337789, 0.2876405 ,
        0.20401864, 0.30229314, 0.94080462, 0.14905823, 0.23914028,
        0.98865669, 0.84330066, 0.52738021, 0.37195082, 0.8278445 ,
        0.32147775, 0.21728075, 0.72050107, 0.66497911, 0.70063198,
        0.03015462, 0.57353865, 0.98137602, 0.41684764, 0.79347122,
        0.15393363, 0.94795841, 0.9649958 , 0.91246068, 0.21337732,
        0.60945612, 0.34562726, 0.1542751 , 0.3793196 , 0.06708658,
        0.93128214, 0.85778244, 0.75525804, 0.90934901, 0.85331578,
        0.06732959, 0.60148106, 0.45098475]])

        b_L = np.zeros(4)

        def flatten(z_):
            flatten_layer = nn.Flatten()
            linear = nn.LinearWithSoftmax(in_dimension=48, out_dimension=4)

            linear.weights = w_L
            linear.biases = b_L

            a_f = flatten_layer(z_)
            a_L = linear(a_f)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))

            layer_L_grad = linear.backward(labels)
            grad = flatten_layer.backward(linear.weights, layer_L_grad)
            return loss, grad

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

        a_L_minus_2 = np.array([  # shape of 6x1x2x2
            [[[1, 3], [0, 1]]],
            [[[1, 2], [0, 1]]],
            [[[1, 2], [0, 2]]],
            [[[1, 2], [2, -1]]],
            [[[1, 2], [0, 1]]],
            [[[1, 2], [-1, -1]]]
        ])
        kernel = np.array([  # shape of 2x1x2x2
            [[[1, 2],
              [0, -1]]],
            [[[0, -1],
             [1, -1]]]
        ])

        b_conv = np.zeros(2)

        w_L = np.array(
            [[1, 0], [1, 1], [0, 0], [1, 0]])  # shape 4x2

        b_L = np.zeros(4)

        def conv_layer(a):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification

            conv = nn.Conv2d(1, 2, 2)
            conv.set_weights(kernel)
            conv.set_biases(b_conv)

            flatten_layer = nn.Flatten()

            linear_softmax = nn.LinearWithSoftmax(4, 2)
            linear_softmax.weights = w_L
            linear_softmax.biases = b_L

            # forward pass
            z = conv(a)
            z_flatten = flatten_layer(z)

            a_L = linear_softmax(z_flatten)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))

            # backward pass
            layer_L_grad = linear_softmax.backward(labels)
            flatten_grad = flatten_layer.backward(linear_softmax.weights, layer_L_grad)
            conv_grad = conv.backward(flatten_grad)

            return loss, conv_grad

        self.gradient_checker_batch_input(conv_layer, a_L_minus_2)  # batch size test

    def test_gradient_2_conv_layers_batch(self):

        a_L_minus_3 = np.array([  # shape 6x1x3x3
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

        kernel_L_minus2 = np.array(  # shape of 1x1x2x2
            [[[[-1, 0], [0, 1]]]])

        b_L_minus_2 = np.zeros(1)  # shape of 1x1

        kernel_L_minus_1 = np.array([[[[-1]]], [[[0]]]])  # shape of 2x1x1x1

        b_L_minus_1 = np.zeros(2)  # shape of 2x1

        w_L = np.array(  # shape of 2x8
            [[1, -1, 0, 1, 1, 0, 0, 0],
             [1, -1, 1, 1, 1, -1, -1, 0]])

        b_L = np.zeros(2)

        def conv_layer(a):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification

            conv1 = nn.Conv2d(1, 1, 2)
            conv1.set_weights(kernel_L_minus2)
            conv1.set_biases(b_L_minus_2)

            conv2 = nn.Conv2d(2, 1, 1)
            conv2.set_weights(kernel_L_minus_1)
            conv2.set_biases(b_L_minus_1)

            flatten_layer = nn.Flatten()

            linear = nn.Linear(2, 2)
            linear.weights = w_L
            linear.biases = b_L

            softmax_layer = nn.Softmax()

            # forward pass
            z_L_minus_2 = conv1(a)
            a_L_minus_2 = z_L_minus_2

            z_L_minus_1 = conv2(a_L_minus_2)
            a_L_minus_1 = z_L_minus_1

            z_flatten = flatten_layer(a_L_minus_1)

            z_L = linear(z_flatten)
            a_L = softmax_layer(z_L)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))

            # backward pass
            layer_L_grad = softmax_layer.backward(labels)
            flatten_grad = flatten_layer.backward(linear.weights, layer_L_grad)
            conv2_grad = conv2.backward(flatten_grad)
            conv1_grad = conv1.backward(conv2_grad)

            return loss, conv1_grad

        self.gradient_checker_batch_input(conv_layer, a_L_minus_3)  # batch size test

    def test_gradient_2_conv_layers_batch_with_sigmoid(self):

        a_L_minus_3 = np.array([  # shape 6x1x3x3
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

        kernel_L_minus2 = np.array(  # shape of 1x1x2x2
            [[[[-1, 0], [0, 1]]]])

        b_L_minus_2 = np.zeros(1)  # shape of 1x1

        kernel_L_minus_1 = np.array([[[[-1]]], [[[0]]]])  # shape of 2x1x1x1

        b_L_minus_1 = np.zeros(2)  # shape of 2x1

        w_L = np.array(  # shape of 2x8
            [[1, -1, 0, 1, 1, 0, 0, 0],
             [1, -1, 1, 1, 1, -1, -1, 0]])

        b_L = np.zeros(2)

        def conv_layer(a):
            """
            the derivative check in the gradient checker relates to the input of the function
            hence, the input should be z - since the backward step computes @loss / @z
            """

            # simulate end of classification

            conv1 = nn.Conv2d(1, 1, 2)
            conv1.set_weights(kernel_L_minus2)
            conv1.set_biases(b_L_minus_2)

            sig1 = nn.Sigmoid()

            conv2 = nn.Conv2d(2, 1, 1)
            conv2.set_weights(kernel_L_minus_1)
            conv2.set_biases(b_L_minus_1)

            sig2 = nn.Sigmoid()

            flatten_layer = nn.Flatten()

            linear = nn.Linear(2, 2)
            linear.weights = w_L
            linear.biases = b_L

            softmax_layer = nn.Softmax()

            # forward pass
            z_L_minus_2 = conv1(a)
            a_L_minus_2 = sig1(z_L_minus_2)

            z_L_minus_1 = conv2(a_L_minus_2)
            a_L_minus_1 = sig2(z_L_minus_1)

            z_flatten = flatten_layer(a_L_minus_1)

            z_L = linear(z_flatten)
            a_L = softmax_layer(z_L)

            labels = np.zeros(a_L.shape)
            labels[:, 1] = 1
            loss = -np.log(np.sum(a_L * labels, axis=1))

            # backward pass
            layer_L_grad = softmax_layer.backward(labels)
            flatten_grad = flatten_layer.backward(linear.weights, layer_L_grad) * sig2.backward()
            conv2_grad = conv2.backward(flatten_grad) * sig1.backward()
            conv1_grad = conv1.backward(conv2_grad)

            return loss, conv1_grad

        self.gradient_checker_batch_input(conv_layer, a_L_minus_3)  # batch size test


if __name__ == '__main__':
    unittest.main()
