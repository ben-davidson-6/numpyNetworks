import numpy as np

from numpy_networks.infrastructure.computation_graph import ComputationGraph
from numpy_networks.infrastructure.graph_layer import GraphLayer
from numpy_networks.infrastructure.layer_func import LayerFunc
from numpy_networks.layer_funcs import convolution
from numpy_networks.layer_funcs import cross_entropy
from numpy_networks.layer_funcs import im2ten
from numpy_networks.layer_funcs import linear
from numpy_networks.layer_funcs import my_max_pool
from numpy_networks.layer_funcs import pad
from numpy_networks.layer_funcs import relu
from numpy_networks.layer_funcs import ten2flat


def get_graph_a(batch_size):
    """
    GRAPH A
    """
    classes = 10

    W = np.random.randn(28 * 28, 10)
    b = np.ones([10]) * 0.01
    params = [W, b]
    output_shape = [batch_size, classes]
    linear_func = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [1]
    cross_entropy_func = LayerFunc(output_shape, cross_entropy)

    linear_layer = GraphLayer(linear_func, 'Linear1')
    cross_entropy_layer = GraphLayer(cross_entropy_func, 'cross_entropy')

    g = {
        linear_layer: cross_entropy_layer
    }

    return ComputationGraph(g, linear_layer, cross_entropy_layer)


def get_graph_b(batch_size):
    """
    GRAPH B
    """
    classes = 10
    sigma = 0.1

    W = np.random.randn(28 * 28, 128) * sigma
    b = np.ones([128]) * 0.01
    params = [W, b]
    output_shape = [batch_size, 128]
    linear_func_1 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [batch_size, 128]
    relu_func = LayerFunc(output_shape, relu)

    W = np.random.randn(128, 10) * sigma
    b = np.ones([10]) * 0.01
    params = [W, b]
    output_shape = [batch_size, 10]
    linear_func_2 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [1]
    cross_entropy_func = LayerFunc(output_shape, cross_entropy.gen_f)

    linear_layer_1 = GraphLayer(linear_func_1, 'Linear1')
    relu_layer = GraphLayer(relu_func, 'relu')
    linear_layer_2 = GraphLayer(linear_func_2, 'Linear2')
    cross_entropy_layer = GraphLayer(cross_entropy_func, 'cross_entropyx')

    g = {
        linear_layer_1: relu_layer,
        relu_layer: linear_layer_2,
        linear_layer_2: cross_entropy_layer
    }

    return ComputationGraph(g, linear_layer_1, cross_entropy_layer)


def get_graph_c(batch_size):
    """
    GRAPH B
    """
    classes = 10
    sigma = 0.001

    W = np.random.randn(28 * 28, 256) * sigma
    b = np.ones([256]) * 0.001
    params = [W, b]
    output_shape = [batch_size, 256]
    linear_func_1 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [batch_size, 256]
    relu_func_1 = LayerFunc(output_shape, relu)

    W = np.random.randn(256, 256) * sigma
    b = np.ones([256]) * 0.001
    params = [W, b]
    output_shape = [batch_size, 256]
    linear_func_2 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [batch_size, 256]
    relu_func_2 = LayerFunc(output_shape, relu)

    W = np.random.randn(256, 10) * sigma
    b = np.ones([10]) * 0.001
    params = [W, b]
    output_shape = [batch_size, 10]
    linear_func_3 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [1]
    cross_entropy_func = LayerFunc(output_shape, cross_entropy)

    linear_layer_1 = GraphLayer(linear_func_1, 'Linear1')
    relu_layer_1 = GraphLayer(relu_func_1, 'relu1')
    linear_layer_2 = GraphLayer(linear_func_2, 'Linear2')
    relu_layer_2 = GraphLayer(relu_func_2, 'relu2')
    linear_layer_3 = GraphLayer(linear_func_3, 'Linear3')
    cross_entropy_layer = GraphLayer(cross_entropy_func, 'cross_entropyx')

    g = {
        linear_layer_1: relu_layer_1,
        relu_layer_1: linear_layer_2,
        linear_layer_2: relu_layer_2,
        relu_layer_2: linear_layer_3,
        linear_layer_3: cross_entropy_layer
    }

    return ComputationGraph(g, linear_layer_1, cross_entropy_layer)


def get_graph_d(batch_size):
    sigma = 0.1
    bias = 0
    channels = 16

    output_shape = [batch_size, 28, 28, 1]
    params = (28, 28)
    train = False
    im2ten_1 = LayerFunc(output_shape, im2ten, params)

    output_shape = [batch_size, 30, 30, 1]
    params = (2, 2)
    train = False
    pad_1 = LayerFunc(output_shape, pad, params)

    output_shape = [batch_size, 28, 28, channels]
    W = np.random.randn(1, 3, 3, channels) * sigma
    b = np.ones([channels]) * bias

    params = (W, b, (3, 3), 1)
    conv_1 = LayerFunc(output_shape, convolution, params, trainable=True)

    output_shape = [batch_size, 14, 14, channels]
    params = (2, 2, 2)
    after_max = LayerFunc(output_shape, my_max_pool, params, max_pool=True)

    output_shape = [batch_size, 16, 16, channels]
    params = (2, 2)
    train = False
    pad_2 = LayerFunc(output_shape, pad, params)

    output_shape = [batch_size, 14, 14, channels]
    W = np.random.randn(channels, 3, 3, channels) * sigma
    b = np.ones([channels]) * bias
    params = (W, b, (3, 3), 1)
    conv_2 = LayerFunc(output_shape, convolution, params, trainable=True)

    output_shape = [batch_size, 7, 7, channels]
    params = (2, 2, 2)
    after_max_2 = LayerFunc(output_shape, my_max_pool, params, max_pool=True)

    output_shape = [batch_size, 7 * 7 * channels]
    params = (7, 7, channels)
    flatten = LayerFunc(output_shape, ten2flat, params, trainable=False)

    W = np.random.randn(7 * 7 * channels, 256) * sigma
    b = np.ones([256]) * bias
    params = [W, b]
    output_shape = [batch_size, 256]
    linear_func_3 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [batch_size, 256]
    relu_func = LayerFunc(output_shape, relu)

    W = np.random.randn(256, 10) * sigma
    b = np.ones([10]) * bias
    params = [W, b]
    output_shape = [batch_size, 10]
    linear_func_4 = LayerFunc(output_shape, linear, params, trainable=True)

    output_shape = [1]
    cross_entropy_func = LayerFunc(output_shape, cross_entropy)

    im2ten_1 = GraphLayer(im2ten_1, 'reshape image')
    pad_1 = GraphLayer(pad_1, 'pad_1')
    conv_1 = GraphLayer(conv_1, 'conv_1')
    after_max = GraphLayer(after_max, 'max_pool 1')
    pad_2 = GraphLayer(pad_2, 'pad2')
    conv_2 = GraphLayer(conv_2, 'conv_2')
    after_max_2 = GraphLayer(after_max_2, 'max_pool 2')
    flatten = GraphLayer(flatten, 'flatten')
    lin_1 = GraphLayer(linear_func_3, 'linear_1')
    relu_1 = GraphLayer(relu_func, 'relu')
    lin_2 = GraphLayer(linear_func_4, 'linear_2')
    cross_entropy_layer = GraphLayer(cross_entropy_func, 'cross_entropyx')

    g = {
        im2ten_1: pad_1,
        pad_1: conv_1,
        conv_1: after_max,
        after_max: pad_2,
        pad_2: conv_2,
        conv_2: after_max_2,
        after_max_2: flatten,
        flatten: lin_1,
        lin_1: relu_1,
        relu_1: lin_2,
        lin_2: cross_entropy_layer
    }

    return ComputationGraph(g, im2ten_1, cross_entropy_layer)
