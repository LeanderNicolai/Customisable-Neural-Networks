from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
from math import e


class Neuron():
    def __init__(self, x, y, bias, layer_id, neuron_id):
        self.x = x
        self.y = y
        self.bias = bias
        self.layer_id = layer_id
        self.neuron_id = neuron_id
        print(f"Created neuron with layer_id {self.layer_id}, and neuron_id {self.neuron_id}")

    def sigmoid(self, x):
        return 1 / (1 + e**(-x))

    def draw(self, neuron_radius):
        neuron_bias = self.bias
        rv_bias = neuron_bias[::-1]
        rev_bias = rv_bias[1:]
        for i in rev_bias:
            print("rev bias shapes are:", i.shape)
        bias_list = [x.T[0].tolist() for x in rev_bias]
        print(bias_list)
        if self.layer_id == 0:
            current_bias = 0.5
        else:
            print(f'trying to slice {self.layer_id} and {self.neuron_id}')
            current_bias = bias_list[self.layer_id - 1][self.neuron_id]
            print('the current bias is: ', current_bias)
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, color='b',
                               fill=False, linewidth=current_bias/3 + 0.2
                               )
        # bc = pyplot.Circle((self.x, self.y), radius=neuron_radius,
        #                    color='b', fill=True, alpha=1)
        pyplot.gca().add_patch(circle)
        # pyplot.gca().add_patch(bc)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights, bias):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.weights = weights
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.layer_id = self.__get_id(network)
        self.bias = bias
        self.neurons = self.__intialise_neurons(number_of_neurons, self.layer_id, self.bias)

    def __intialise_neurons(self, number_of_neurons, layer_id, bias):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, bias, layer_id, iteration)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __get_id(self, network):
        return len(network.layers)

    def __line_between_two_neurons(self, neuron1, neuron2, weight):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=weight)
        print(neuron1.x, neuron1.y, neuron2.x, neuron2.y, " Weight is: ", weight)
        pyplot.gca().add_line(line)

    def sigmoid(self, x):
        return 1 / (1 + e**(-x))

    def draw(self, layerType=0):
        weights = self.weights
        weight_index = self.layer_id - 1
        print('length of neurons in this layer is', len(self.neurons))
        if weight_index == -1:
            weight_index = 0
        print("\n\n")
        print(f"Length of Weights in layer {weight_index} is: ", len(weights[weight_index]))
        print(f"Length of self.neurons in layer {weight_index} is ", len(self.neurons))
        print("\n\n")
        if self.previous_layer:
            for neuron, layer_weight in zip(self.neurons, weights[weight_index]):
                neuron.draw(self.neuron_radius)
                if self.previous_layer:
                    for previous_layer_neuron, weight in zip(self.previous_layer.neurons, layer_weight):
                        print("drawing a line with weight", abs(weight))
                        self.__line_between_two_neurons(
                            neuron, previous_layer_neuron, self.sigmoid(weight) * 2)
        else:
            for neuron in self.neurons:
                neuron.draw(self.neuron_radius)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize=12)


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, weights, bias):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.weights = weights
        self.layers = []
        self.layertype = 0
        self.bias = bias

    def add_layer(self, number_of_neurons, weights):
        layer = Layer(self, number_of_neurons,
                      self.number_of_neurons_in_widest_layer, self.weights, self.bias)
        self.layers.append(layer)

    def draw(self):
        fig = pyplot.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        pyplot.draw()
        # pyplot.pause(0)  # <-------
        # input("<Hit Enter To Close>")
        return fig


class DrawNN():
    def __init__(self, neural_network, weights, bias):
        self.neural_network = neural_network
        self.weights = weights
        self.bias = bias

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer, self.weights, self.bias)
        for l in self.neural_network:
            network.add_layer(l, self.weights)
        fig = network.draw()
        return fig
