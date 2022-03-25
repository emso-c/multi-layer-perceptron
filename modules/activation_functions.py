from abc import ABCMeta, abstractmethod
import math


class ActivationFunction(object):
    __metaclass__ = ABCMeta
    
    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def apply(self, value) -> float:
        pass

class Sigmoid(ActivationFunction):
    name = "Logistic"

    def apply(value) -> float:
        return 1 / (1 + math.exp(-value))

class ReLU(ActivationFunction):
    name = "Rectified Linear Activation"

    def apply(value) -> float:
        return max(0, value)

class TanH(ActivationFunction):
    name = "Hyperbolic Tangent"

    def apply(value) -> float:
        return math.tanh(value)

class Linear(ActivationFunction):
    name = "Linear"

    def apply(value) -> float:
        return value

class ActivationFunctions:
    SIGMOID = Sigmoid
    RELU = ReLU
    TANH = TanH
    LINEAR = Linear
