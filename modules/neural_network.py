from typing import Generator, Tuple, Union
from abc import ABCMeta

from .layer import Layer

class NeuralNetwork(object):
    __metaclass__ = ABCMeta

class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self, layers:list[Layer], weight_range:Tuple[float, float]=(-1.0, 1.0)):
        self.layers = layers
        self.weight_range = weight_range

        self.activate(randomize_input_weights=True)

    def __repr__(self):
        return "<class=MultiLayerPerceptron>\n\tInput Layer = {}\n\tHidden Layers = {}\n\tOutput Layer = {}\n".format(
            self.input_layer,
            self.hidden_layers,
            self.output_layer,
        )

    @property
    def input_layer(self) -> list[int]:
        return self.layers[0]

    @property
    def hidden_layers(self) -> list[Union[Layer, None]]:
        return self.layers[1:-1]
    
    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    def activate(self, randomize_input_weights=False) -> None:
        prev_layer = None
        for layer in self.layers[1:]:
            if prev_layer is None:
                input_data = self.input_layer
            else:
                input_data = [perceptron.output() for perceptron in prev_layer.perceptrons]
            for perceptron in layer.perceptrons:
                perceptron.input_data = input_data
                if randomize_input_weights:
                    perceptron.randomize_input_weights(self.weight_range)
            prev_layer = layer

    def output(self) -> Generator[float, None, None]:
        for perceptron in self.output_layer.perceptrons:
            yield perceptron.output()
