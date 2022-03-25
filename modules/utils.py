import random
from sys import maxsize
from typing import Generator, Union

def generate_random_input_data(
        n,
        precision:Union[int, None]=None,
        min_val:float=-maxsize,
        max_val:float=maxsize
    ) -> Generator[float, None, None]:

    for _ in range(n):
        yield random.uniform(min_val, max_val)

def normalize_dataset(dataset:list[float], min_scale:float=0, max_scale:float=10) -> list:
    if len(dataset) <= 1: return dataset

    min_val, max_val = min(dataset), max(dataset)
    if min_val == max_val: return dataset

    normalized_dataset = []
    for num in dataset:
        normalized_dataset.append(
            ((num-min_val)/(max_val-min_val))*(max_scale-min_scale)+min_scale
        )
    return normalized_dataset

def subscript(num):
    return ''.join([subscript_digit(int(d)) for d in str(num)])

def subscript_digit(digit) -> str:
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