from typing import Tuple

from .perceptron import Perceptron
from .activation_functions import ActivationFunction, Sigmoid
from .summation_functions import SummationFunction, WeightedTotal

class Layer:
    def __init__(self, neurons:list[Perceptron]):
        self.neurons = neurons

    def add_neuron(self, neuron:Perceptron):
        self.neurons.append(neuron)

    def remove_neuron_by_index(self, i:int):
        del self.neurons[i]

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, index):
        return self.neurons[index]

    def __repr__(self):
        return f"Layer object with {len(self.neurons)} neurons"

def generate_random_layer(
        layer_depth:int,
        activation_function:ActivationFunction=Sigmoid,
        summation_function:SummationFunction=WeightedTotal,
        normalization_scale:Tuple=(0,1),
        bias:float=1,
    ) -> Layer:

    layer = Layer([])
    for _ in range(layer_depth):
        layer.add_neuron(Perceptron(
            activation_function=activation_function,
            normalization_scale=normalization_scale,
            summation_function=summation_function,
            bias=bias,
        ))
    return layer
