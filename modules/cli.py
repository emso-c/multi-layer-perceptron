import os
from .perceptron import Perceptron
from .activation_functions import ActivationFunctions
from .utils import subscript, generate_random_input_data


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

        neuron = Perceptron(
            input_data=input_data,
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
