import os
import random

from modules.perceptron import Perceptron
from modules.neural_network import MultiLayerPerceptron
from modules.layer import Layer, generate_random_layer
from modules.activation_functions import ActivationFunctions
from modules.utils import subscript, generate_random_input_data

def single_neuron_creation_CLI():
    while True:
        os.system("cls")
        n = int(input("Girdi sayisini giriniz: "))
        min_val = float(input("Minimum girdi degerini giriniz: "))
        max_val = float(input("Maximum girdi degerini giriniz: "))
        min_weight = float(input("Minimum girdi agirlik katsayisini giriniz: "))
        max_weight = float(input("Maximum girdi agirlik katsayisini giriniz: "))
        min_scale = float(input("Minimum ölçütleme degerini giriniz: "))
        max_scale = float(input("Maximum ölçütleme degerini giriniz: "))

        print("Aktivasyon fonksiyonu seciniz:\n1-ReLU\n2-Sigmoid\n3-TanH")
        choice = input("Seciminiz: ")
        if choice == 1:
            activation_function = ActivationFunctions.SIGMOID
        elif choice == 2:
            activation_function = ActivationFunctions.RELU
        elif choice == 3:
            activation_function = ActivationFunctions.TANH
        else:
            print("Gecersiz giris, sigmoid ile devam ediliyor...")
            activation_function = ActivationFunctions.SIGMOID

        input_data = list(generate_random_input_data(
            n=n,
            min_val=min_val,
            max_val=max_val,
        ))

        neuron = Neuron(
            input_values=input_data,
            activation_function=activation_function,
            normalization_scale=(min_scale, max_scale)
        )
        neuron.randomize_input_weights(min_weight, max_weight)

        print("\nOlusturulan veri seti:")
        for i, data in enumerate(input_data):
            print(f"X{subscript(i)}: {data},\t W{subscript(i)}: {neuron.weights[i]}")
        
        print("\nOlceklendirilmis veri seti:")
        for i, data in enumerate(neuron.normalized_inputs):
            print(f"X{subscript(i)}: {data},\t W{subscript(i)}: {neuron.weights[i]}")

        print("\nAgirlikli deger toplami: ", neuron.weighted_input_summary)

        print("\nCikti: ", neuron.output())

        res = input("\nDevam etmek istiyor musunuz? (e/h): ")
        if res != "e":
            break

if __name__ == '__main__':

    # single_neuron_creation_CLI()
    # exit()

    INPUT_AMOUNT = 5
    LAYER_AMOUNT = 3
    ACTIVATION_FUNCTION = ActivationFunctions.TANH
    NORMALIZATION_SCALE = (-1,1)
    MIN_LAYER_DEPTH, MAX_LAYER_DEPTH = 1, 3
    MIN_INPUT_VALUE, MAX_INPUT_VALUE = -1000000, 1000000
    MIN_WEIGHT_VALUE, MAX_WEIGHT_VALUE = -1, 1

    inputs = list(generate_random_input_data(
        n=INPUT_AMOUNT,
        min_val=MIN_INPUT_VALUE,
        max_val=MAX_INPUT_VALUE,
    ))


    input_layer = generate_random_layer(
        layer_depth=len(inputs),
        activation_function=ACTIVATION_FUNCTION,
        normalization_scale=NORMALIZATION_SCALE,
    )
    layers = [generate_random_layer(
        layer_depth=random.randint(MIN_LAYER_DEPTH, MAX_LAYER_DEPTH),
        activation_function=ACTIVATION_FUNCTION,
        normalization_scale=NORMALIZATION_SCALE,
    ) for _ in range(LAYER_AMOUNT-1)] # -1 because input layer is already defined

    nn = MultiLayerPerceptron(
        inputs = inputs,
        layers = [input_layer] + layers,
        min_weight=MIN_WEIGHT_VALUE,
        max_weight=MAX_WEIGHT_VALUE
    )

    print(nn)
    print("Layers:")
    print("Inputs:", inputs)
    print("Input Layer:", nn.input_layer)
    print("Hidden Layers:")
    for hidden_layer in nn.hidden_layers:
        print(hidden_layer)
        for neuron in hidden_layer.neurons:
            print(neuron)
    print("Output Layer:")
    print(nn.output_layer)
    for neuron in nn.output_layer:
        print(neuron)
    
