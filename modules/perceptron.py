import random
from dataclasses import dataclass
from typing import Tuple

from .activation_functions import ActivationFunction, Sigmoid
from .utils import normalize_dataset

@dataclass
class Perceptron:
    def __init__(self, input_values:list[float]=[], weights:list[float]=[], activation_function:ActivationFunction=Sigmoid, normalization_scale:Tuple=(0,1)):
        self.input_data = input_values # TODO make immutable
        self.weights = weights # TODO make immutable
        self.activation_function = activation_function
        self.normalization_scale = normalization_scale

    def __str__(self):
        return f"""\t<class=Neuron>
            Inputs: {self.input_data}
            Weights: {self.weights}
            Activation function: {self.activation_function.name}
            Normalization Scale: {self.normalization_scale}
            Output: {self.output()}
        """

    def add_input(self, value, weight):
        self.input_data.append(value)
        self.weights.append(weight)

    def randomize_input_weights(self, min_weight:float=-1, max_weight:float=1) -> list[float]:
        self.weights = [random.uniform(min_weight, max_weight) for _ in range(len(self.input_data))]
        return self.weights

    @property
    def weighted_input_summary(self):
        sum=0
        for i in range(len(self.normalized_inputs)):
            sum+=self.normalized_inputs[i]*self.weights[i]
        return sum

    @property
    def normalized_inputs(self):
        return normalize_dataset(
            self.input_data,
            self.normalization_scale[0],
            self.normalization_scale[1]
        )

    def output(self) -> float:
        return self.activation_function.apply(self.weighted_input_summary)
