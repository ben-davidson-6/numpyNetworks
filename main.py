from numpy_networks import train_graphs, get_results

# can be 'linear', '1_hidden', '2_hidden', or 'convolution'
# will display training and validation accuracy graph when complete
# training finishes after either 200000 steps or after a plataeu of
# 7 steps
train_graphs.train('convolution')

# will run the saved models on the test set
# get_results.get_results()
