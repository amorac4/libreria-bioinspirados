from __future__ import annotations
import numpy as np
from typing import Callable
from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ===========================================
# Whale Swarm Algorithm (WSA) - Vectorizado
# Basado en Zeng et al. (2017)
# ===========================================
def wsa(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        rho0: float = 2.0,     # Intensidad inicial
        eta: float = 0.0,      # Coeficiente de atenuación
        log_positions: bool = False, log_every: int = 1):
    
    # 1. Inicialización
    # ------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Posiciones iniciales
    X = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # Evaluamos la población inicial
    fitness = objective(X)
    
    # Inicializar el mejor global
    best_idx = np.argmin(fitness)
    G = X[best_idx].copy()
    Gf = fitness[best_idx]
    
    # --- HISTORIAL (Formato exacto de GWO) ---
    # Se inicializan todas las claves que usa tu visualizador
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest", "pos", "gbest_hist"))
    
    # Guardar estado inicial (t=0)
    hist["best_f"].append(Gf)
    hist["mean_f"].append(np.mean(fitness))
    hist["best_x"].append(G.copy())
    hist["gbest"].append(G.copy())
    
    if log_positions:
        hist["pos"].append(X.copy())
        hist["gbest_hist"].append(G.copy())

    # 2. Bucle de Optimización
    # -------------------------
    for it in range(max_iters):
        
        
        # 1. Matriz de distancias
        diff_matrix = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff_matrix**2, axis=2))

        # 2. Identificar vecinos mejores
        better_mask = fitness[None, :] < fitness[:, None]
        
        # 3. Encontrar el vecino mejor MÁS CERCANO
        masked_dists = np.where(better_mask, dists, np.inf)
        target_indices = np.argmin(masked_dists, axis=1)
        min_dists = np.min(masked_dists, axis=1)
        valid_target = np.isfinite(min_dists)
        
        # 4. Definir objetivo Y
        Y = np.where(valid_target[:, np.newaxis], X[target_indices], X)
        d_XY = np.where(valid_target, min_dists, 0.0)[:, np.newaxis]

        # 5. Movimiento (rho y actualización)
        rho = rho0 * np.exp(-eta * d_XY)
        rand_vec = rng.random((pop_size, dim))
        
        X_new = X + rand_vec * rho * (Y - X)
        clamp_vec_np(X_new, bounds, in_place=True)
        
        # --- B. Evaluación ---
        fit_new = objective(X_new)
        
        # Actualización de población
        X = X_new
        fitness = fit_new
        
        # --- C. Actualizar Mejor Global ---
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < Gf:
            Gf = fitness[current_best_idx]
            G = X[current_best_idx].copy()
            
        # --- D. Guardar Historial ---
        hist["best_f"].append(Gf)
        hist["mean_f"].append(np.mean(fitness))
        hist["best_x"].append(G.copy())
        hist["gbest"].append(G.copy())
        
        if log_positions and ((it + 1) % log_every == 0):
            hist["pos"].append(X.copy())        
            hist["gbest_hist"].append(G.copy())    
            
    return G, Gf, hist