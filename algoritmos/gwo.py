from __future__ import annotations
import numpy as np
from typing import Callable
from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ===========================================
# Grey Wolf Optimizer (GWO) - NumPy Vectorizado
# Basado en Mirjalili et al. (2014)
# ===========================================
def gwo(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        log_positions: bool = False, log_every: int = 1):
    """
    Implementación del Grey Wolf Optimizer (GWO).
    Jerarquía: Alpha (mejor), Beta (2do), Delta (3ro), Omega (resto).
    """
    
    # 1. Inicialización
    # ------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Posiciones iniciales de los lobos (Omegas)
    X = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # Evaluamos la población inicial
    fitness = objective(X)
    
    # Inicializamos Alpha, Beta y Delta con valores infinitos (para minimización)
    # Se actualizarán inmediatamente antes del bucle
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    # Función auxiliar para actualizar la jerarquía
    def update_hierarchy(X_pop, fit_pop):
        nonlocal Alpha_pos, Alpha_score, Beta_pos, Beta_score, Delta_pos, Delta_score
        
        # Ordenamos por fitness (menor es mejor)
        sorted_indices = np.argsort(fit_pop)
        
        # Iteramos sobre la población ordenada para actualizar los líderes
        for i in sorted_indices:
            f = fit_pop[i]
            pos = X_pop[i]
            
            if f < Alpha_score:
                # El antiguo Alpha pasa a ser Beta, Beta a Delta...
                # (Pero en la implementación estándar de GWO, simplemente 
                #  verificamos contra los scores actuales para reasignar)
                # Aquí usamos la lógica estricta de comparación:
                
                Alpha_score = f
                Alpha_pos = pos.copy()
                
            elif f < Beta_score and f > Alpha_score: # Evitar duplicar si hay empate exacto
                Beta_score = f
                Beta_pos = pos.copy()
                
            elif f < Delta_score and f > Beta_score and f > Alpha_score:
                Delta_score = f
                Delta_pos = pos.copy()
    
    # Primera actualización de la jerarquía
    update_hierarchy(X, fitness)

    # Historial
    hist_keys = ["best_f", "mean_f", "best_x", "gbest"]
    if log_positions:
        hist_keys.extend(["pos", "gbest_hist"])
    hist = init_history(keys=hist_keys)

    # 2. Bucle Principal
    # ------------------
    for t in range(max_iters):
        
        # 'a' disminuye linealmente de 2 a 0
        a = 2.0 - (2.0 * t / max_iters)
        
        # --- Vectorización de las Ecuaciones de Caza ---
        # X(t+1) = (X1 + X2 + X3) / 3
        # Donde X1 se basa en Alpha, X2 en Beta, X3 en Delta
        
        # Generamos r1 y r2 para toda la población y dimensiones
        # Shape: (pop_size, dim)
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        
        # A = 2*a*r1 - a
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        # D_alpha = |C1 * Alpha_pos - X|
        # X1 = Alpha_pos - A1 * D_alpha
        D_alpha = np.abs(C1 * Alpha_pos - X)
        X1 = Alpha_pos - A1 * D_alpha
        
        # Repetimos para Beta
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        D_beta = np.abs(C2 * Beta_pos - X)
        X2 = Beta_pos - A2 * D_beta
        
        # Repetimos para Delta
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        D_delta = np.abs(C3 * Delta_pos - X)
        X3 = Delta_pos - A3 * D_delta
        
        # --- Actualización de Posición ---
        X_new = (X1 + X2 + X3) / 3.0
        
        # Limitar a los bordes
        clamp_vec_np(X_new, bounds, in_place=True)
        
        # --- Evaluación ---
        fit_new = objective(X_new)
        
        # En GWO estándar, los lobos SIEMPRE se mueven.
        # No hay selección "greedy" (como en DE) para mantener la posición vieja.
        # Los lobos siguen a los líderes incondicionalmente.
        X = X_new
        fitness = fit_new
        
        # --- Actualizar Jerarquía (Alpha, Beta, Delta) ---
        update_hierarchy(X, fitness)
        
        # --- Guardar Historial ---
        hist["best_f"].append(Alpha_score)
        hist["mean_f"].append(np.mean(fitness))
        hist["best_x"].append(Alpha_pos.copy())
        hist["gbest"].append(Alpha_pos.copy())
        
        if log_positions and (t % log_every == 0):
            hist["pos"].append(X.copy())
            hist["gbest_hist"].append(Alpha_pos.copy())

    return Alpha_pos, Alpha_score, hist