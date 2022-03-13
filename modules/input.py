from dataclasses import dataclass

@dataclass
class Input:
    value:float
    weight:float

    @property
    def weighted_value(self):
        return self.value*self.weight
    
    def __repr__(self) -> str:
        return f"{self.value} ({self.weight})"

    def __add__(self, i2) -> float:
        return self.value + i2.value

    def __sub__(self, i2) -> float:
        return self.value + i2.value