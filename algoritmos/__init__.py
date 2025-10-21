
from .pso import pso
from .ga import ga
from .de import de
from .aco import aco

ALGORITMOS_REGISTRADOS = {
    "pso": pso,
    "ga": ga,
    "de": de,
    "aco": aco
 
}

__all__ = [
    "pso",
    "ga",
    "de",
    "aco",
    "ALGORITMOS_REGISTRADOS"
]