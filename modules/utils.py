from .input import Input
import random
from sys import maxsize
from typing import Union

def generate_random_input_data(
        n,
        precision:Union[int, None]=None,
        min_val:float=-maxsize,
        max_val:float=maxsize,
        min_weight:float=-1,
        max_weight:float=1
    ) -> list[Input]:

    for _ in range(n):
        value = random.uniform(min_val, max_val)
        weight = random.uniform(min_weight, max_weight)
        yield Input(value=value, weight=weight)

def randomize_input_weights(input_data, min_weight:float=-1, max_weight:float=1):
    for data in input_data:
        data.weight = random.uniform(min_weight, max_weight)
    return input_data

def normalize_dataset(dataset, min_scale:int=0, max_scale:int=100) -> list:
    normalized_dataset = []
    min_val, max_val = min(dataset), max(dataset)
    for num in dataset:
        normalized_dataset.append(
            ((num-min_val)/(max_val-min_val))*(max_scale-min_scale)+min_scale
        )
    return normalized_dataset

def subscript(num):
    return ''.join([subscript_digit(int(d)) for d in str(num)])

def subscript_digit(digit):
    if digit == 0:
        return u"\u2080"
    elif digit == 1:
        return u"\u2081"
    elif digit == 2:
        return u"\u2082"
    elif digit == 3:
        return u"\u2083"
    elif digit == 4:
        return u"\u2084"
    elif digit == 5:
        return u"\u2085"
    elif digit == 6:
        return u"\u2086"
    elif digit == 7:
        return u"\u2087"
    elif digit == 8:
        return u"\u2088"
    elif digit == 9:
        return u"\u2089"
    raise ValueError