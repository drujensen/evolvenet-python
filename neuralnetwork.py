import random
import math
from typing import List
from network import Network


class Synapse:
    def __init__(self, index: int, weight: float = 0.0):
        self.index = index
        self.weight = weight

    def clone(self):
        return Synapse(self.index, self.weight)

    def randomize(self):
        self.weight = random.uniform(-1.0, 1.0)

    def mutate(self, rate: float):
        self.weight += random.uniform(-rate, rate)

    def punctuate(self, pos: int):
        self.weight = round(self.weight, pos)

    def to_json(self):
        return {"i": self.index, "w": self.weight}


class Neuron:
    def __init__(self, function: str):
        self.synapses: List[Synapse] = []
        self.function = function
        self.activation = 0.0
        self.bias = 0.0

    def clone(self):
        neuron = Neuron(self.function)
        neuron.activation = self.activation
        neuron.bias = self.bias
        neuron.synapses = [synapse.clone() for synapse in self.synapses]
        return neuron

    def randomize(self):
        self.bias = random.uniform(-1.0, 1.0)
        for synapse in self.synapses:
            synapse.randomize()

    def mutate(self, rate: float):
        self.bias += random.uniform(-rate, rate)
        synapse_rate = rate / len(self.synapses)
        for synapse in self.synapses:
            synapse.mutate(synapse_rate)

    def punctuate(self, pos: int):
        self.bias = round(self.bias, pos)
        for synapse in self.synapses:
            synapse.punctuate(pos)

    def set(self, value: float):
        self.activation = self.none(value)

    def activate(self, input_data: 'Layer'):
        parent = input_data
        if self.function == "min":
            minimum = parent.neurons[self.synapses[0].index].activation
            for synapse in self.synapses:
                minimum = min(minimum, parent.neurons[synapse.index].activation)
            self.activation = self.none(minimum)
        elif self.function == "avg":
            total = sum(parent.neurons[synapse.index].activation for synapse in self.synapses)
            self.activation = self.none(total / len(self.synapses))
        elif self.function == "max":
            maximum = max(parent.neurons[synapse.index].activation for synapse in self.synapses)
            self.activation = self.none(maximum)
        else:
            total = sum(synapse.weight * parent.neurons[synapse.index].activation for synapse in self.synapses)
            if self.function == "none":
                self.activation = self.none(total + self.bias)
            elif self.function == "relu":
                self.activation = self.relu(total + self.bias)
            elif self.function == "sigmoid":
                self.activation = self.sigmoid(total + self.bias)
            elif self.function == "tanh":
                self.activation = self.tanh(total + self.bias)
            else:
                raise ValueError(f"Activation function {self.function} is not supported")

    def none(self, value):
        return float(value)

    def relu(self, value):
        return max(0.0, value)

    def sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(-value))

    def tanh(self, value):
        return math.tanh(value)

    def to_json(self):
        return {
            "s": [synapse.to_json() for synapse in self.synapses],
            "f": self.function,
            "b": self.bias
        }


class Layer:
    def __init__(self, layer_type: str, size: int = 0, function: str = "sigmoid"):
        self.type = layer_type
        self.size = size
        self.function = function
        self.neurons = [Neuron(function) for _ in range(self.size)]

    def clone(self):
        layer = Layer(self.type, 0, self.function)
        layer.neurons = [neuron.clone() for neuron in self.neurons]
        return layer

    def randomize(self):
        for neuron in self.neurons:
            neuron.randomize()

    def mutate(self, rate: float):
        neuron_rate = rate / len(self.neurons)
        for neuron in self.neurons:
            neuron.mutate(neuron_rate)

    def punctuate(self, pos: int):
        for neuron in self.neurons:
            neuron.punctuate(pos)

    def set(self, values: List[float]):
        for neuron, val in zip(self.neurons, values):
            neuron.set(val)

    def activate(self, parent: 'Layer'):
        for neuron in self.neurons:
            neuron.activate(parent)

    def to_json(self):
        return {
            "t": self.type,
            "n": [neuron.to_json() for neuron in self.neurons],
            "s": self.size,
            "f": self.function
        }


class NeuralNetwork(Network):
    def __init__(self):
        self.layers: List[Layer] = []
        self.error = 1.0

    def add_layer(self, layer_type: str, size: int = None, function: str = "sigmoid"):
        self.layers.append(Layer(layer_type, size=size, function=function))

    def clone(self):
        network = NeuralNetwork()
        network.layers = [layer.clone() for layer in self.layers]
        return network

    def randomize(self):
        self.error = 1.0
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            layer.randomize()
        return self

    def mutate(self):
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            layer.mutate(self.error)
        return self

    def punctuate(self, pos: int):
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            layer.punctuate(pos)
        return self

    def run(self, data: List[float]):
        for index, layer in enumerate(self.layers):
            if index == 0:
                layer.set(data)
            else:
                layer.activate(self.layers[index - 1])
        return [neuron.activation for neuron in self.layers[-1].neurons]

    def evaluate(self, data: List[List[List[float]]]):
        sum_error = 0.0
        for input_data, expected in data:
            actual = self.run(input_data)
            for act, exp in zip(actual, expected):
                sum_error += (exp - act) ** 2
        self.error = sum_error / (2 * len(data))

    def fully_connect(self):
        for index, layer in enumerate(self.layers):
            if index == 0:
                continue
            parent = self.layers[index - 1]
            for neuron in layer.neurons:
                for idx in range(len(parent.neurons)):
                    neuron.synapses.append(Synapse(idx))

    def to_json(self):
        return {
            "l": [layer.to_json() for layer in self.layers],
            "e": self.error
        }
