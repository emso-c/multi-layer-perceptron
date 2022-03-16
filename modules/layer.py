from typing import Tuple

from .perceptron import Perceptron
from .activation_functions import ActivationFunction, Sigmoid

class Layer:
    def __init__(self, neurons:list[Perceptron]):
        self.neurons = neurons

    def add_neuron(self, neuron:Perceptron):
        self.neurons.append(neuron)

    def remove_neuron_by_index(self, i:int):
        del self.neurons[i]

    def __getitem__(self, index):
        return self.neurons[index]

    def __repr__(self):
        return f"Layer object with {len(self.neurons)} neurons"

class InputLayer:
    def __init__(self, inputs:list[int]):
        self.inputs = inputs

    def __repr__(self):
        return f"InputLayer object with {len(self.inputs)} inputs"

def generate_random_layer(
        layer_depth:int,
        activation_function:ActivationFunction=Sigmoid,
        normalization_scale:Tuple=(0,1),
    ) -> Layer:

    layer = Layer([])
    for _ in range(layer_depth):
        layer.add_neuron(Perceptron(
            activation_function=activation_function,
            normalization_scale=normalization_scale
        ))
    return layer
