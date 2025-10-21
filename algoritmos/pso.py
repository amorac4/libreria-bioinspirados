from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# PSO completo (global-best) - NumPy
# ==================================
def pso(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
        vmax: float | None = None,
        log_positions: bool = False, log_every: int = 1):
    """ Algoritmo de Optimización por Enjambre de Partículas (PSO), versión NumPy. """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed) # Generador de números aleatorios
    dim = bounds.shape[0]             # Dimensión del problema
    
    # --- Inicialización Vectorizada ---
    X = rand_vec_in_bounds_np(bounds, pop_size, rng) # Posiciones
    V = np.zeros((pop_size, dim))                     # Velocidades
    P = np.copy(X)                                    # Mejores Posiciones Personales (PBest)
    
    # *** OPTIMIZACIÓN ***
    # Evalúa la población entera de una sola vez.
    Pf = objective(P)                                 # Mejores Fitness Personales (PBest Fitness)
    
    # --- Mejor Global ---
    g_idx = np.argmin(Pf) # Índice del mejor
    G = P[g_idx].copy()   # Mejor Posición Global (GBest)
    Gf = Pf[g_idx]        # Mejor Fitness Global (GBest Fitness)

    # Diccionario para guardar el historial de convergencia
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = [] # Guarda posiciones para animar

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        # --- Actualizaciones vectorizadas de velocidad y posición ---
        r1, r2 = rng.random((pop_size, dim)), rng.random((pop_size, dim))
        
        # Ecuación de velocidad de PSO (vectorizada)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        
        if vmax is not None:
            np.clip(V, -vmax, vmax, out=V) # Limita la velocidad
        
        # Ecuación de posición
        X += V
        clamp_vec_np(X, bounds, in_place=True) # Limita las posiciones
        
        # --- Evaluación ---
        # *** OPTIMIZACIÓN ***
        # Evalúa todas las nuevas posiciones de una sola vez
        f_vals = objective(X)
        
        # --- Actualizar mejores personales ---
        update_indices = f_vals < Pf # Índices booleanos de quién mejoró
        Pf[update_indices] = f_vals[update_indices]
        P[update_indices] = X[update_indices]
        
        # --- Actualizar mejor global ---
        g_idx = np.argmin(Pf) # Encuentra el mejor PBest actual
        if Pf[g_idx] < Gf:
            Gf = Pf[g_idx] # Nuevo récord global
            G = P[g_idx].copy()

        # --- Guardando en historial ---
        hist["best_f"].append(Gf)
        hist["mean_f"].append(np.mean(f_vals))
        hist["best_x"].append(G.copy())
        hist["gbest"].append(G.copy())
        if log_positions and dim == 2 and (it % log_every == 0):
            hist["pos"].append(X.copy())

    # 3. Fin
    # --------
    return G, Gf, hist