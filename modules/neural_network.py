from typing import Generator, Union
from .layer import InputLayer, Layer

class NeuralNetwork():
    def __init__(self, layers:list[Union[InputLayer,Layer]], min_weight:float=-1, max_weight:float=1):
        self.layers = layers
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.activate(randomize_input_weights=True)

    def __repr__(self):
        return f"""
        <class=NeuralNetwork>
            Input Layer = {self.input_layer}
            Hidden Layers = {self.hidden_layers}
            Output Layer = {self.output_layer}
        """

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
                input_data = self.input_layer.inputs
            else:
                input_data = [neuron.output() for neuron in prev_layer.neurons]

            for neuron in layer.neurons:
                neuron.input_data = input_data
                if randomize_input_weights:
                    neuron.randomize_input_weights(self.min_weight, self.max_weight)
            prev_layer = layer

    def output(self) -> Generator[float, None, None]:
        for neuron in self.output_layer.neurons:
            yield neuron.output()
