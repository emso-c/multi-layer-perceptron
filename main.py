import random
import os
from sys import maxsize

from modules.neuron import Neuron
from modules.activation_functions import ReLU, Sigmoid, TanH
from modules.input import Input

from modules.utils import subscript 

def generate_rand_inp_data(n, min_val:float=-maxsize, max_val:float=maxsize, min_weight:float=-1, max_weight:float=1) -> list[Input]:
    for _ in range(n):
        yield Input(
            value=random.uniform(min_val, max_val),
            weight=random.uniform(min_weight, max_weight)
        )

def cli():
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
            activation_function = Sigmoid
        elif choice == 2:
            activation_function = ReLU
        elif choice == 3:
            activation_function = TanH
        else:
            print("Gecersiz giris, sigmoid ile devam ediliyor...")
            activation_function = Sigmoid

        input_data = list(generate_rand_inp_data(
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
            scale=(min_scale, max_scale)
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
    cli()
