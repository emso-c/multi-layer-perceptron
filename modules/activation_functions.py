from abc import ABCMeta, abstractmethod
import math


class ActivationFunction(object):
    __metaclass__ = ABCMeta
    
    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def apply(self, value):
        pass

class Sigmoid(ActivationFunction):
    name = "Logistic"

    def apply(value):
        return 1 / (1 + math.exp(-value))

class ReLU(ActivationFunction):
    name = "Rectified Linear Activation"

    def apply(value):
        return max(0, value)

class TanH(ActivationFunction):
    name = "Hyperbolic Tangent"

    def apply(value):
        return math.tanh(value)
