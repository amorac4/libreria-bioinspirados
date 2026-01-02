from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ================================
# GA real-coded (funcional) - NumPy
# ================================
def ga(objective: Callable, bounds: np.ndarray,
       max_iters: int, pop_size: int, seed: int | None = None,
       pc: float = 0.9, pm: float = 0.05, elitism: int = 1,
       tournament_k: int = 3, blx_alpha: float = 0.3,
       sigma_frac: float = 0.1,
       log_positions: bool = False, log_every: int = 1):
    """ Algoritmo Genético (GA) para variables reales, versión NumPy. """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    assert pop_size >= max(2, elitism + 1)

    # Genera la población inicial
    pop = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # *** OPTIMIZACIÓN ***
    # Evalúa la población entera de una sola vez
    fit = objective(pop)
    
    # Guarda el mejor individuo inicial
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_f = fit[best_idx]

    # Prepara el historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = []
        
    # Rangos de los límites (para la mutación)
    ranges = bounds[:, 1] - bounds[:, 0]

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        # --- Elitismo ---
        # Los 'n' mejores individuos pasan directo a la siguiente generación
        elite_indices = np.argsort(fit)[:elitism]
        new_pop = pop[elite_indices].copy()

        # Bucle para crear el resto de la nueva generación
        while len(new_pop) < pop_size:
            # --- Selección por Torneo ---
            # Elige k individuos al azar para 2 torneos (para 2 padres)
            p_indices = rng.choice(pop_size, (2, tournament_k), replace=False)
            # Compara el fitness de los seleccionados
            tourn_fit = fit[p_indices]
            # Obtiene los índices de los 2 ganadores
            winner_indices = p_indices[np.arange(2), np.argmin(tourn_fit, axis=1)]
            p1, p2 = pop[winner_indices] # Padre 1 y Padre 2

            # --- Cruce BLX-alpha ---
            if rng.random() < pc:
                alpha = rng.uniform(-blx_alpha, 1 + blx_alpha, size=(2, dim))
                c1 = alpha[0] * p1 + (1 - alpha[0]) * p2
                c2 = alpha[1] * p1 + (1 - alpha[1]) * p2
                children = np.vstack([c1, c2])
                clamp_vec_np(children, bounds, in_place=True) # Limita a los hijos
            else:
                children = np.vstack([p1, p2]) # Los hijos son clones

            # --- Mutación Gaussiana ---
            # Crea una máscara de booleanos para ver qué genes mutar
            mutate_mask = rng.random((len(children), dim)) < pm
            if np.any(mutate_mask):
                # Genera ruido gaussiano
                mutations = rng.normal(0.0, sigma_frac * ranges, size=children.shape)
                # Aplica el ruido solo donde la máscara es True
                children[mutate_mask] += mutations[mutate_mask]
                clamp_vec_np(children, bounds, in_place=True) # Limita de nuevo
            
            # Añade los hijos a la nueva población
            new_pop = np.vstack([new_pop, children])

        # --- Reemplazo ---
        pop = new_pop[:pop_size] # La nueva generación reemplaza a la antigua
        
        # *** OPTIMIZACIÓN ***
        # Evalúa la nueva población entera de una sola vez
        fit = objective(pop)

        # --- Actualizar mejor global ---
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