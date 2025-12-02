from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ============================================================
# Mean Particle Swarm Optimization (MeanPSO) - NumPy
# Basado en Deep & Bansal (2009)
# ============================================================
def mean_pso(objective: Callable, bounds: np.ndarray,
             max_iters: int, pop_size: int, seed: int | None = None,
             w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
             vmax: float | None = None,
             log_positions: bool = False, log_every: int = 1):
    """
    MeanPSO: Las partículas son atraídas por el promedio (Mean) 
    entre su pbest y el gbest, en lugar de ser atraídas por ellos por separado.
    """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Inicialización Vectorizada
    X = rand_vec_in_bounds_np(bounds, pop_size, rng) # Posiciones
    V = np.zeros((pop_size, dim))                     # Velocidades
    P = np.copy(X)                                    # Mejores Posiciones Personales (PBest)
    
    # Evaluación inicial
    Pf = objective(P)                                 # Mejores Fitness Personales
    
    # Mejor Global (GBest)
    g_idx = np.argmin(Pf)
    G = P[g_idx].copy()
    Gf = Pf[g_idx]

    # Historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "std_f", "gbest", "pos", "gbest_hist"))
    
    # Guardar estado inicial
    hist["best_f"].append(Gf)
    hist["mean_f"].append(np.mean(Pf))
    hist["std_f"].append(np.std(Pf))
    hist["best_x"].append(G.copy())
    hist["gbest"].append(G.copy())
    if log_positions:
        hist["pos"].append(X.copy())
        hist["gbest_hist"].append(G.copy())
    
    # 2. Bucle de Optimización
    # --------------------------
    for it in range(max_iters):
        
        # --- CÁLCULO DEL MEAN (La clave de MeanPSO) ---
        # Mean = (Pbest + Gbest) / 2
        # Broadcasting: P es (pop, dim), G es (dim,). NumPy maneja la suma correctamente.
        Meanp = (P + G) / 2.0
        Meanm = (P - G) / 2.0
        
        # --- Actualización de Velocidad y Posición ---
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        
        # Ecuación modificada de MeanPSO:
        # Ambos términos (cognitivo y social) apuntan hacia 'Mean' en lugar de P y G
        V = w * V + c1 * r1 * (Meanp - X) + c2 * r2 * (Meanm - X)
        
        # Limitar velocidad (si vmax está definido)
        if vmax is not None:
            np.clip(V, -vmax, vmax, out=V)
        
        # Actualizar Posición
        X += V
        clamp_vec_np(X, bounds, in_place=True)
        
        # --- Evaluación ---
        f_vals = objective(X)
        
        # --- Actualizar PBest ---
        update_indices = f_vals < Pf
        Pf[update_indices] = f_vals[update_indices]
        P[update_indices] = X[update_indices]
        
        # --- Actualizar GBest ---
        min_p_idx = np.argmin(Pf)
        if Pf[min_p_idx] < Gf:
            Gf = Pf[min_p_idx]
            G = P[min_p_idx].copy()

        # --- Guardar Historial ---
        hist["best_f"].append(Gf)
        hist["mean_f"].append(np.mean(f_vals))
        hist["std_f"].append(np.std(f_vals))
        hist["best_x"].append(G.copy())
        hist["gbest"].append(G.copy())
        
        if log_positions and (it + 1) % log_every == 0:
            hist["pos"].append(X.copy())
            hist["gbest_hist"].append(G.copy())

    return G, Gf, hist