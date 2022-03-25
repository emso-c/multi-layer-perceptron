import random

from modules.neural_network import MultiLayerPerceptron
from modules.layer import generate_random_layer
from modules.utils import generate_random_input_data

from config import *

from modules.cli import single_neuron_creation_CLI

if __name__ == '__main__':

    #single_neuron_creation_CLI()

    input_layer = list(generate_random_input_data(
        n=INPUT_AMOUNT,
        min_val=MIN_INPUT_VALUE,
        max_val=MAX_INPUT_VALUE,
    ))

    layers = [generate_random_layer(
        layer_depth=random.randint(MIN_LAYER_DEPTH, MAX_LAYER_DEPTH),
        activation_function=ACTIVATION_FUNCTION,
        summation_function=SUMMATION_FUNCTION,
        normalization_scale=NORMALIZATION_SCALE,
        bias=BIAS,
    ) for _ in range(LAYER_AMOUNT)]

    mlp = MultiLayerPerceptron(
        layers = [input_layer] + layers,
        weight_range=WEIGHT_RANGE
    )

    print(mlp)
    print("Inputs:", input_layer)
    print("Layers:\n")
    print("Hidden Layers:\n")
    for hidden_layer in mlp.hidden_layers:
        print(str(hidden_layer)+"\n")
        for perceptron in hidden_layer.perceptrons:
            print(perceptron)
    print(f"Output Layer: {str(mlp.output_layer)}\n")
    for perceptron in mlp.output_layer:
        print(perceptron)
