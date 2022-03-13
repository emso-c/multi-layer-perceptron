from dataclasses import dataclass
from typing import Tuple

from modules.activation_functions import ActivationFunction, Sigmoid
from .input import Input
from .utils import normalize_dataset

@dataclass
class Neuron:
    def __init__(self, input_data:list[Input], activation_function:ActivationFunction=Sigmoid, scale:Tuple=(0,1)):
        self.input_data = input_data
        self.activation_function = activation_function
        self.scale = scale

    @property
    def input_sum(self):
        sum = 0
        for data in self.normalized_input:
            sum = sum + data.weighted_value
        return sum
    
    @property
    def normalized_input(self):
        dataset = [data.value for data in self.input_data]
        normalized_dataset = normalize_dataset(dataset, self.scale[0], self.scale[1])
        norm_inp = []
        for i in range(len(normalized_dataset)):
            norm_inp.append(Input(
                value=normalized_dataset[i],
                weight=self.input_data[i].weight
            ))
        return norm_inp

    def output(self):
        return self.activation_function.apply(self.input_sum)