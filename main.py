import random

from modules.neural_network import MultiLayerPerceptron
from modules.layer import Layer, generate_random_layer
from modules.activation_functions import ActivationFunctions
from modules.summation_functions import SummationFunctions
from modules.utils import generate_random_input_data

from modules.cli import single_neuron_creation_CLI

if __name__ == '__main__':

    #single_neuron_creation_CLI()

    INPUT_AMOUNT = 5
    LAYER_AMOUNT = 3
    BIAS = 1
    ACTIVATION_FUNCTION = ActivationFunctions.TANH
    SUMMATION_FUNCTION = SummationFunctions.WEIGHTED_TOTAL
    NORMALIZATION_SCALE = (-1,1)
    MIN_LAYER_DEPTH, MAX_LAYER_DEPTH = 1, 3
    MIN_INPUT_VALUE, MAX_INPUT_VALUE = -1000000, 1000000
    MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE = -1, 1

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
        min_weight=MIN_WEIGHT_VALUE,
        max_weight=MAX_WEIGHT_VALUE
    )

    print(mlp)
    print("Inputs:", input_layer)
    print("Layers:\n")
    print("Hidden Layers:\n")
    for hidden_layer in mlp.hidden_layers:
        print(str(hidden_layer)+"\n")
        for neuron in hidden_layer.neurons:
            print(neuron)
    print(f"Output Layer: {str(mlp.output_layer)}\n")
    for neuron in mlp.output_layer:
        print(neuron)
