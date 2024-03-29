from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.weights = weights
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.layer_id = self.__get_id(network)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
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

    def draw(self, layerType=0):
        weights = self.weights
        weight_index = self.layer_id - 1
        sample_weights = [[[1, 1, 1], [1, 1, 1], [3, 3, 3]], [[1, 1, 1], [2, 4, 2], [1, 1, 1]]]
        for neuron, layer_weight in zip(self.neurons, sample_weights[weight_index]):
            print("Printing lengths", neuron, layer_weight)
            print("We are in layer", self.layer_id)
            print("Weight index is", weight_index)
            print("Weights to be drawn are", sample_weights[weight_index])
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron, weight in zip(self.previous_layer.neurons, layer_weight):
                    print("drawing a line with weight", weight)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, float(weight))
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize=12)


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, weights):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.weights = weights
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weights):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, self.weights)
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
        pyplot.suptitle('Press enter to start training', fontsize=10)
        pyplot.title('Neural Network architecture', fontsize=15)
        pyplot.draw()
        pyplot.pause(1)  # <-------
        input("<Hit Enter To Close>")
        pyplot.close(fig)


class DrawNN():
    def __init__(self, neural_network, weights):
        self.neural_network = neural_network
        self.weights = weights

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer, weights)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()

    def give_network():
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        return network


# b = [3, 3, 3]
# NN = DrawNN(b)
# NN.draw()

# NN.give_network()
