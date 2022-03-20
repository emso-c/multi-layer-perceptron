import random
from dataclasses import dataclass
from typing import Tuple

from .activation_functions import ActivationFunction, Sigmoid
from .summation_functions import SummationFunction, WeightedTotal
from .utils import normalize_dataset

@dataclass
class Perceptron:
    def __init__(self,
            input_data:list[float]=[],
            weights:list[float]=[],
            activation_function:ActivationFunction=Sigmoid,
            summation_function:SummationFunction=WeightedTotal,
            normalization_scale:Tuple=(0,1),
            bias:float=1
        ):
        self.input_data = input_data # TODO make immutable
        self.weights = weights # TODO make immutable
        self.activation_function = activation_function
        self.summation_function = summation_function
        self.normalization_scale = normalization_scale
        self.bias = bias

    def __str__(self):
        return "<class=Perceptron>\n\tInputs: {}\n\tWeights: {}\n\tActivation function: {}\n\tSummation function: {}\n\tNormalization Scale: {}\n\tBias: {}\n\tOutput: {}\n".format(
            self.input_data,
            self.weights,
            self.activation_function.name,
            self.summation_function.name,
            self.normalization_scale,
            self.bias,
            self.output(),
        )

    def add_input(self, value, weight):
        self.input_data.append(value)
        self.weights.append(weight)

    def randomize_input_weights(self, weight_range:Tuple[float, float]=(-1.0, 1.0)) -> list[float]:
        self.weights = [random.uniform(weight_range[0], weight_range[1]) for _ in range(len(self.input_data))]
        return self.weights

    @property
    def normalized_inputs(self):
        return normalize_dataset(
            self.input_data,
            self.normalization_scale[0],
            self.normalization_scale[1]
        )
    
    @property
    def summation(self):
        return self.summation_function.apply(
            inputs = self.normalized_inputs,
            weights = self.weights
        )

    def output(self) -> float:
        return self.activation_function.apply(self.summation+self.bias)
