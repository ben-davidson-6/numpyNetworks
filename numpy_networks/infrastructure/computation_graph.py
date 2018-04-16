import numpy as np

class ComputationGraph:

    def __init__(self, graph, input_layer, final_layer):
        self.graph = graph
        self.reverse_graph = {v: k for k, v in graph.items()}
        self.input_layer = input_layer
        self.final_layer = final_layer

    def forward_pass(self, inputs, labels):
        """
        Args:
            inputs = array of images
            labels = array of labels

        Note:
            Calculate from input to the network
            all the way to output. Also saves
            the current input of a layer which
            is used in the backward pass
        """
        inputs = self.input_layer.output(inputs)
        current_layer = self.input_layer

        while True:
            if self.graph[current_layer] is self.final_layer:
                # if labels is none, then we are using it for inference
                output = inputs if labels is None else self.final_layer.output((labels, inputs))
                return output
            inputs = self.graph[current_layer].output(inputs)
            current_layer = self.graph[current_layer]
        return output

    def back_pass(self, learn):
        """
        Args:
            learn = float learning rate

        Note:
            Starting from final layer calculate
            dlayer/dinput and send this to the
            next layer. Multiply the backproped
            gradient by this layers gradients.
        """

        # currently considered layer
        layer = self.final_layer

        # This phantom dim will always be None!
        product, phantom_dim = self.final_layer.layer_func.d_dinput()
        while True:
            if self.reverse_graph[layer] is self.input_layer:
                product = self.input_layer.backprop(product, learn)
                break
            product = self.reverse_graph[layer].backprop(product, learn)
            layer = self.reverse_graph[layer]
        return product

    def train(self, inputs, labels, learn):
        self.forward_pass(inputs, labels)
        self.back_pass(learn, )

    def predict(self, data):
        """Run some data through the network and get back a softmax score"""
        scores = self.forward_pass(data, None)
        scores = np.exp(scores)
        return scores / scores.sum(axis=1)[:, None]

    def set_params(self, params):
        """set the parameters of the graph"""
        layer = self.input_layer
        i = 0
        if layer.is_trainable():
            layer.set_params(params[i])
            i = 1
            if len(params) == 1:
                return

        while True:
            if self.graph[layer] is self.final_layer:
                if self.final_layer.is_trainable():
                    self.final_layer.set_params(params[i])
                break
            layer = self.graph[layer]
            if layer.is_trainable():
                layer.set_params(params[i])
                i += 1

    def get_params(self, ):
        """returns a list of parameters ordered in the same order of the graph"""
        layer = self.input_layer
        params = []
        if layer.is_trainable():
            params.append(layer.get_params())
        while True:
            if self.graph[layer] is self.final_layer:
                break
            layer = self.graph[layer]
            if layer.is_trainable():
                params.append(layer.get_params())
        return params
