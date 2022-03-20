from ..modules.perceptron import Perceptron
from ..modules.summation_functions import SummationFunctions

inputs = [0.5, 0.6, 0.2, 0.7]
weights = [-0.2, 0.6, 0.2, -0.1]
perceptron = Perceptron(
    input_data = inputs,
    weights = weights,
    summation_function=SummationFunctions.WEIGHTED_TOTAL,
    normalization_scale=(min(inputs), max(inputs))
)

print(perceptron.summation)
print(perceptron.output())