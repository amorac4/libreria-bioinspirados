from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ================================
# DE/rand/1/bin (funcional) - NumPy
# ================================
def de(objective: Callable, bounds: np.ndarray,
       max_iters: int, pop_size: int, seed: int | None = None,
       F: float = 0.5, CR: float = 0.9,
       log_positions: bool = False, log_every: int = 1):
    """ Algoritmo de Evolución Diferencial (DE), versión NumPy. """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    assert pop_size >= 4, "DE requiere al menos 4 individuos"

    # Genera la población inicial
    pop = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # *** OPTIMIZACIÓN ***
    # Evalúa la población inicial entera de una sola vez
    fit = objective(pop)
    
    # Guarda el mejor inicial
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_f = fit[best_idx]

    # Prepara el historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = []

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        
        # Itera sobre cada individuo de la población
        for i in range(pop_size):
            
            # --- Mutación (DE/rand/1) ---
            # Selección de 3 individuos distintos (a, b, c)
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]

            # Crea el vector mutante: v = a + F * (b - c)
            v = a + F * (b - c) 

            # --- Cruce Binomial ---
            # Máscara booleana para el cruce
            cross_points = rng.random(dim) < CR
            # Garantiza que al menos un gen mute (jrand)
            jrand = rng.integers(dim)
            cross_points[jrand] = True
            
            # Crea el vector de prueba 'u'
            u = np.where(cross_points, v, pop[i])
            clamp_vec_np(u, bounds, in_place=True) # u es un solo vector
            
            # --- Selección ---
         
            fu = objective(u) # Evalúa solo al vector de prueba
            
            # Si el nuevo vector 'u' es mejor, reemplaza al individuo 'i'
            if fu <= fit[i]:
                pop[i] = u
                fit[i] = fu
        
        # --- Actualizar mejor global ---
        # (Después de que toda la población ha sido procesada)
        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < best_f:
            best_f = fit[current_best_idx]
            best_x = pop[current_best_idx].copy()

        # --- Guardando en historial ---
        hist["best_f"].append(best_f)
        hist["mean_f"].append(np.mean(fit))
        hist["best_x"].append(best_x.copy())
        hist["gbest"].append(best_x.copy())
        if log_positions and dim == 2 and (it % log_every == 0):
            hist["pos"].append(pop.copy())
            
    # 3. Fin
    # --------
    return best_x, best_f, hist