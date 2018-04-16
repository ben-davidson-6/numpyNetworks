import numpy as np


def predictions(graph, images, labels, j, batch_size):
    preds = graph.predict(images[j * batch_size: (j + 1) * batch_size, :])
    predicted = np.argmax(preds, axis=1)
    actual = np.argmax(labels[j * batch_size: (j + 1) * batch_size], axis=1)
    return predicted, actual


def count_acc(graph, images, labels, batch_size):

    confusion_mat = np.zeros([10, 10]).astype(np.int32)
    total_correct = 0.0
    num_images = images.shape[0]

    for j in range(num_images // batch_size):
        predicted, actual = predictions(graph, images, labels, j, batch_size)
        total_correct += np.sum(predicted == actual)
        confusion_mat[actual, predicted] += 1
    acc = total_correct / num_images

    return acc, confusion_mat