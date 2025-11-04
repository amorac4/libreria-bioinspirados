
from .pso import pso
from .ga import ga
from .de import de
from .aco import aco
from .bat import bat
from .cso import cso

ALGORITMOS_REGISTRADOS = {
    "pso": pso,
    "ga": ga,
    "de": de,
    "aco": aco,
    "bat": bat,
    "cso": cso
}

__all__ = [
    "pso",
    "ga",
    "de",
    "aco",
    "bat",
    "cso",
    "ALGORITMOS_REGISTRADOS"
]