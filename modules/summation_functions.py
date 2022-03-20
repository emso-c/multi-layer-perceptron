from abc import ABCMeta, abstractmethod
from math import prod


class SummationFunction(object):
    __metaclass__ = ABCMeta
    
    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def apply(self, inputs:list[float], weights:list[float]) -> float:
        pass

class WeightedTotal(SummationFunction):
    name = "Weighted Total"

    def apply(inputs:list[float], weights:list[float]) -> float:
        assert len(weights) == len(inputs)
        return sum([inputs[i]*weights[i] for i in range(len(inputs))])


class Multiplication(SummationFunction):
    name = "Multiplication"

    def apply(inputs:list[float], weights:list[float]) -> float:
        assert len(weights) == len(inputs)
        return prod([inputs[i]*weights[i] for i in range(len(inputs))])


class Maximum(SummationFunction):
    name = "Maximum"

    def apply(inputs:list[float], weights:list[float]) -> float:
        assert len(weights) == len(inputs)
        return max([inputs[i]*weights[i] for i in range(len(inputs))])

class Minimum(SummationFunction):
    name = "Minimum"

    def apply(inputs:list[float], weights:list[float]) -> float:
        assert len(weights) == len(inputs)
        return min([inputs[i]*weights[i] for i in range(len(inputs))])

class SummationFunctions:
    WEIGHTED_TOTAL = WeightedTotal
    MULTIPLICATION = Multiplication
    MAXIMUM = Maximum
    MINIMUM = Minimum
