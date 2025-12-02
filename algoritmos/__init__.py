
from .pso import pso
from .ga import ga
from .de import de
from .aco import aco
from .bat import bat
from .cso import cso
from .c_so import c_so
from .fwa import fwa
from .gwo import gwo
from .wsa import wsa
from .mean_pso import mean_pso

ALGORITMOS_REGISTRADOS = {
    "pso": pso,
    "ga": ga,
    "de": de,
    "aco": aco,
    "bat": bat,
    "cso": cso,
    "c_so": c_so,
    "fwa": fwa,
    "gwo": gwo,
    "wsa": wsa,
	"mean_pso": mean_pso
}

__all__ = [
    "pso",
    "ga",
    "de",
    "aco",
    "bat",
    "cso",
    "c_so",
    "fwa",
    "gwo",
    "wsa",
	"mean_pso",
    "ALGORITMOS_REGISTRADOS"
]