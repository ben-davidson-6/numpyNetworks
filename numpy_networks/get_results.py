import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

from numpy_networks.utils.misc import *
from .graphs import *


def get_results():
    """Show classification accuracy over the 4 networks"""

    # get graph builders and data
    builders = [get_graph_a, get_graph_b, get_graph_c, get_graph_d]
    mnist = input_data.read_data_sets(os.path.join(os.getcwd(), 'data'), one_hot=True)
    test_img, test_lbl = mnist.test.images, mnist.test.labels
    tr_img, tr_lbl = mnist.train.images, mnist.train.labels
    batch_size = 64

    for builder in builders:

        # load parameter file
        model_file = os.path.join(os.getcwd(), 'models', builder.__name__ + '.pickle')
        with open(model_file, 'rb') as handle:
            obj = pickle.load(handle)
            params = obj['params']

        # build graph and fill with params
        graph = builder(batch_size)
        graph.set_params(params)

        # test graph performance
        test_acc, confusion_mat = count_acc(graph, test_img, test_lbl, batch_size)
        train_acc, _ = count_acc(graph, tr_img, tr_lbl, batch_size)

        # display performance
        print(builder.__name__)
        print(confusion_mat)
        print('test acc  %.3f' % test_acc)
        print('train acc %.3f' % train_acc)
        print('*' * 100)
