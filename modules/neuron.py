from dataclasses import dataclass
from typing import Tuple

from modules.activation_functions import ActivationFunction, Sigmoid
from .input import Input
from .utils import normalize_dataset

@dataclass
class Neuron:
    def __init__(self, input_data:list[Input], activation_function:ActivationFunction=Sigmoid, normalization_scale:Tuple=(0,1)):
        self.input_data = input_data
        self.activation_function = activation_function
        self.normalization_scale = normalization_scale

    @property
    def input_sum(self):
        sum = 0
        for data in self.normalized_input:
            sum = sum + data.weighted_value
        return sum
    
    @property
    def normalized_input(self):
        dataset = [data.value for data in self.input_data]
        normalized_dataset = normalize_dataset(dataset, self.normalization_scale[0], self.normalization_scale[1])
        norm_inp = []
        for i in range(len(normalized_dataset)):
            norm_inp.append(Input(
                value=normalized_dataset[i],
                weight=self.input_data[i].weight
            ))
        return norm_inp
    
    def __str__(self):
        return f"""<class=Neuron>
        Input amount: {len(self.input_data)}
        Activation function: {self.activation_function.name}
        Normalization Scale: {self.normalization_scale}
        """
    
    def report(self) -> str:
        return f"""
        <class=Neuron>
        Input amount: {len(self.input_data)}
        Input summary: {self.input_sum}
        Activation function: {self.activation_function.name}
        Normalization Scale: {self.normalization_scale}
        """

    def output(self) -> float:
        return self.activation_function.apply(self.input_sum)