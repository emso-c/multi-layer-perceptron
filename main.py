import os
import random

from modules.input import Input
from modules.neuron import Neuron
from modules.activation_functions import ActivationFunctions

from modules.utils import randomize_input_weights, subscript, generate_random_input_data

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
            min_weight=min_weight,
            max_weight=max_weight
        ))

        print("\nOlusturulan veri seti:")
        for i, data in enumerate(input_data):
            print(f"X{subscript(i)}: {data.value},\t W{subscript(i)}: {data.weight}")
        
        neuron = Neuron(
            input_data=input_data,
            activation_function=activation_function,
            normalization_scale=(min_scale, max_scale)
        )

        print("\nOlceklendirilmis veri seti:")
        for i, data in enumerate(neuron.normalized_input):
            print(f"X{subscript(i)}: {data.value},\t W{subscript(i)}: {data.weight}")

        print("\nAgirlikli deger toplami: ", neuron.input_sum)

        print("\nCikti: ", neuron.output())

        res = input("\nDevam etmek istiyor musunuz? (e/h): ")
        if res != "e":
            break

if __name__ == '__main__':
    
    # generate random input

    INPUT_AMOUNT = 3
    DATA_PER_INPUT = 5
    DATA_PRECISION = None

    MAX_INPUT_VALUE, MIN_INPUT_VALUE = -10000, 10000
    MIN_INPUT_WEIGHT, MAX_INPUT_WEIGHT = -1, 1

    input_layer = list(generate_random_input_data(
        n=DATA_PER_INPUT,
        precision=DATA_PRECISION,
        min_val=MIN_INPUT_VALUE,
        max_val=MAX_INPUT_VALUE,
        min_weight=MIN_INPUT_WEIGHT,
        max_weight=MAX_INPUT_WEIGHT
    ))

    print("\nInput layer:")
    [print(input_data) for input_data in input_layer]

    # generate hidden layers

    HIDDEN_LAYER_AMOUNT = 2
    NEURON_PER_HIDDEN_LAYER = 4

    ACTIVATION_FUNCTION = ActivationFunctions.SIGMOID
    NORMALIZATION_SCALE = (-1, 1)
    
    hidden_layers = []

    for i in range(HIDDEN_LAYER_AMOUNT):
        if i == 0:
            input_data = input_layer
        else:
            input_data = [Input(neuron.output(), 0) for neuron in hidden_layers[-1]]
            randomize_input_weights(input_data)

        print("Adding new layer with inputs:")
        [print(data) for data in input_data]       
        
        layer = []
        for neuron in range(NEURON_PER_HIDDEN_LAYER):
            layer.append(Neuron(
                input_data=input_data,
                activation_function=random.choice([
                    ActivationFunctions.RELU,
                    ActivationFunctions.SIGMOID,
                    ActivationFunctions.SIGMOID
                ]),
                normalization_scale=NORMALIZATION_SCALE
            ))
            print("Neuron added:", layer[-1].report())
        hidden_layers.append(layer)

    """ for hidden_layer in range(HIDDEN_LAYER_AMOUNT):
        layer = []
        for neuron in range(NEURON_PER_HIDDEN_LAYER):
            layer.append(Neuron(
                input_data=input_layer,
                activation_function=ACTIVATION_FUNCTION,
                normalization_scale=NORMALIZATION_SCALE
            ))
        hidden_layers.append(layer) """


    sum = 0
    for neuron in hidden_layers[-1]:
        sum += neuron.output()

    print("\nFinal output: ", sum)

    #single_neuron_creation_CLI()
