
from .pso import pso
from .ga import ga
from .de import de
from .aco import aco
from .bat import bat

ALGORITMOS_REGISTRADOS = {
    "pso": pso,
    "ga": ga,
    "de": de,
    "aco": aco,
    "bat": bat
}

__all__ = [
    "pso",
    "ga",
    "de",
    "aco",
    "bat",
    "ALGORITMOS_REGISTRADOS"
]