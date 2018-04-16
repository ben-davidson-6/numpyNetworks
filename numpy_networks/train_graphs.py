import os
import pickle

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from numpy_networks import graphs
from numpy_networks.utils.misc import *


def train(graph_name):

    # get model builders
    graph_dict = {
        'linear':graphs.get_graph_a,
        '1_hidden':graphs.get_graph_b,
        '2_hidden':graphs.get_graph_c,
        'convolution':graphs.get_graph_d}

    # get data
    mnist = input_data.read_data_sets(os.path.join(os.getcwd(), 'data'), one_hot=True)

    # various params
    tr = []
    val = []
    best_acc = 0
    learn = 0.01
    batch_size = 64
    max_training_steps = 20000
    training_acc_sample_size = 1000
    early_stopping_step_number = 7

    # get graph
    graph = graph_dict[graph_name](batch_size)

    for i in range(max_training_steps):

        # get batch and train
        xx, yy = mnist.train.next_batch(batch_size)
        graph.train(xx, yy, learn)

        # reduce learning rate
        if i != 0 and i % 10000 == 0:
            learn *= 0.1

        # Check accuracy
        if i % 200 == 0:

            # validation accuracy
            val_img, val_lbl = mnist.validation.images, mnist.validation.labels
            val_acc, _ = count_acc(graph, val_img, val_lbl, batch_size)
            print('curr acc %.3f' % val_acc)
            print('best acc %.3f' % best_acc)
            val.append(val_acc)
            del val_img
            del val_lbl

            # Get training accuracy
            tr_img, tr_lbl = mnist.train.next_batch(training_acc_sample_size)
            tr_acc, _ = count_acc(graph, tr_img, tr_lbl, batch_size)
            tr.append(tr_acc)
            del tr_img
            del tr_lbl

            # Early stopping conditions
            if val_acc <= best_acc:
                plateu += 1
                if plateu > early_stopping_step_number:
                    break
            # if model better then save this
            else:
                params = graph.get_params()
                best_acc = val_acc
                with open(os.path.join(os.getcwd(), 'models', graph_dict[graph_name].__name__ + '.pickle'), 'wb') as handle:
                    pickle.dump({'params': params}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                plateu = 0

    # display training, validation curves
    ax = plt.subplot(1, 1, 1)
    ax.plot(range(len(tr)), tr, label='Training')
    ax.plot(range(len(val)), val, label='Validation')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.show()

