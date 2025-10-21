from __future__ import annotations
import numpy as np

def clamp_vec_np(x: np.ndarray, bounds: np.ndarray, in_place: bool = False):
    """ Restringe un vector o matriz de vectores a los límites dados. """
    if not in_place:
        x = x.copy()
    
    # bounds[:, 0] es el vector de mínimos
    # bounds[:, 1] es el vector de máximos
    if x.ndim == 1:
        np.clip(x, bounds[:, 0], bounds[:, 1], out=x)
    elif x.ndim == 2:
        np.clip(x, bounds[:, 0], bounds[:, 1], out=x)
    return x

def rand_vec_in_bounds_np(bounds: np.ndarray, n_vecs: int, rng: np.random.Generator):
    """ Genera n_vecs vectores aleatorios dentro de los límites. """
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low, high, size=(n_vecs, len(bounds)))

def init_history(keys):
    return {k: [] for k in keys}