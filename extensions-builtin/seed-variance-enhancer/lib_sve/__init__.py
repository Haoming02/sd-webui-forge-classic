from enum import Enum
from math import cos, pi
from typing import Callable


class DecayMethod(Enum):
    NA = "No Decay"
    LIN = "Linear"
    COS = "Cosine"
    EXP = "Exponential"
    QUAD = "Quadratic"

    @staticmethod
    def choices() -> list[str]:
        return [item.value for item in DecayMethod]

    @staticmethod
    def decay_function(method: str) -> Callable[[int, int, float], float]:

        match method:
            case DecayMethod.NA.value:
                return lambda current_step, total_steps, strength: strength
            case DecayMethod.LIN.value:
                return lambda current_step, total_steps, strength: strength * (1.0 - current_step / total_steps)
            case DecayMethod.COS.value:
                return lambda current_step, total_steps, strength: strength * cos((current_step / total_steps) * (pi / 2))
            case DecayMethod.EXP.value:
                return lambda current_step, total_steps, strength: strength * (1.0 - (current_step / total_steps)) ** 2
            case DecayMethod.QUAD.value:
                return lambda current_step, total_steps, strength: strength * (1.0 - (current_step / total_steps) ** 2)
