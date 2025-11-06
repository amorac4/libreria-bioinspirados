
from .pso import pso
from .ga import ga
from .de import de
from .aco import aco
from .bat import bat
from .cso import cso
from .c_so import c_so

ALGORITMOS_REGISTRADOS = {
    "pso": pso,
    "ga": ga,
    "de": de,
    "aco": aco,
    "bat": bat,
    "cso": cso,
    "c_so": c_so
}

__all__ = [
    "pso",
    "ga",
    "de",
    "aco",
    "bat",
    "cso",
    "c_so",
    "ALGORITMOS_REGISTRADOS"
]