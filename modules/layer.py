from typing import Tuple

from .perceptron import Perceptron
from .activation_functions import ActivationFunction, Sigmoid
from .summation_functions import SummationFunction, WeightedTotal

class Layer:
    def __init__(self, perceptrons:list[Perceptron], bias:float=1):
        self.perceptrons = perceptrons
        self.bias = bias

        self.update_biases()
    
    def update_biases(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.bias = self.bias

    def add_perceptron(self, perceptron:Perceptron):
        self.perceptrons.append(perceptron)

    def remove_perceptron_by_index(self, i:int):
        del self.perceptrons[i]

    def __len__(self):
        return len(self.perceptrons)

    def __getitem__(self, index):
        return self.perceptrons[index]

    def __repr__(self):
        return f"Layer object with {len(self.perceptrons)} perceptrons"

def generate_random_layer(
        layer_depth:int,
        activation_function:ActivationFunction=Sigmoid,
        summation_function:SummationFunction=WeightedTotal,
        normalization_scale:Tuple=(0,1),
        bias:float=1,
    ) -> Layer:

    layer = Layer([])
    for _ in range(layer_depth):
        layer.add_perceptron(Perceptron(
            activation_function=activation_function,
            normalization_scale=normalization_scale,
            summation_function=summation_function,
            bias=bias,
        ))
    return layer
