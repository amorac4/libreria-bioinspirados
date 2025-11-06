from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Algoritmo de Optimización por Enjambre de Gatos (CSO)
# ==================================
def cso(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        mix_rate: float = 0.5,           # MR: Probabilidad de estar en modo de búsqueda
        seeking_memory_pool: int = 5,    # SMP: Número de puntos candidatos en modo de búsqueda
        seeking_range_factor: float = 0.2, # SRC: Factor para el rango de búsqueda
        velocity_limit_factor: float = 0.03, # VL: Factor para el límite de velocidad en modo de rastreo
        log_positions: bool = False, log_every: int = 1):
   
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed) # Generador de números aleatorios
    dim = bounds.shape[0] # Dimensionalidad del problema
    X = rand_vec_in_bounds_np(bounds, pop_size, rng) # Posiciones iniciales
    V = np.zeros((pop_size, dim))  # Velocidades iniciales en cero
    F = objective(X) # Fitness de cada gato
    g_idx = np.argmin(F) # Índice del mejor gato
    G = X[g_idx].copy() # Mejor posición global
    Gf = F[g_idx]       # Mejor fitness global
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest")) # Historial de la optimización
    if log_positions and dim == 2:
        hist["pos"] = []
    
    # Calcular el rango total del espacio de búsqueda para definir los límites de velocidad
    # (max_bound - min_bound) / 2
    bounds_range = bounds[:, 1] - bounds[:, 0]
    velocity_limit = bounds_range * velocity_limit_factor

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        
        # Decide qué gatos están en modo de búsqueda (seeking) y cuáles en modo de rastreo (tracing)
        # Una decisión para cada gato
        is_seeking_mode = rng.random(pop_size) < mix_rate
        
        # --- MODO DE BÚSQUEDA (Seeking Mode) ---
        # Los gatos "descansan" y miran a su alrededor
        
        seeking_cats_indices = np.where(is_seeking_mode)[0]
        num_seeking_cats = len(seeking_cats_indices)
        
        if num_seeking_cats > 0:
            current_seeking_positions = X[seeking_cats_indices] # Posiciones actuales de estos gatos
            
            # Genera múltiples candidatos para cada gato en modo de búsqueda
            # Cada candidato se genera alrededor de la posición actual del gato
            # El rango es un factor del rango total de los límites
            seeking_range = bounds_range * seeking_range_factor # Vector (dim,) 
            displacements = rng.uniform(-1, 1, (num_seeking_cats, seeking_memory_pool, dim)) * seeking_range # (num_seeking_cats, seeking_memory_pool, dim)
            candidate_positions = current_seeking_positions[:, np.newaxis, :] + displacements 
            
            # Asegura que los candidatos estén dentro de los límites
            for j in range(num_seeking_cats):
                clamp_vec_np(candidate_positions[j], bounds, in_place=True)
            
            # Evalúa todos los candidatos
            # Reshape para evaluar todos a la vez (num_seeking_cats * seeking_memory_pool, dim)
            flat_candidates = candidate_positions.reshape(-1, dim) 
            flat_fitness = objective(flat_candidates) 
            reshaped_fitness = flat_fitness.reshape(num_seeking_cats, seeking_memory_pool) 
            best_candidate_indices_per_cat = np.argmin(reshaped_fitness, axis=1) # Índices de los mejores candidatos por gato
            
            # ========== INICIO DE LA CORRECCIÓN ==========
            
            # Obtener el fitness y la posición de los mejores candidatos encontrados
            best_new_fitness = reshaped_fitness[np.arange(num_seeking_cats), best_candidate_indices_per_cat]
            best_new_positions = candidate_positions[np.arange(num_seeking_cats), best_candidate_indices_per_cat]

            current_fitness = F[seeking_cats_indices]
            improvement_mask = best_new_fitness < current_fitness
            update_indices_local = np.where(improvement_mask)[0]
            
            # Si hay gatos que mejorar, actualizar solo esos
            if update_indices_local.size > 0:

                update_indices_global = seeking_cats_indices[update_indices_local]
                X[update_indices_global] = best_new_positions[update_indices_local]
                F[update_indices_global] = best_new_fitness[update_indices_local]

            # ========== FIN DE LA CORRECCIÓN ==========


        # --- MODO DE RASTREO (Tracing Mode) ---
        # Los gatos "cazan" y se mueven hacia el mejor global
        
        # Selecciona los gatos en modo de rastreo
        tracing_cats_indices = np.where(~is_seeking_mode)[0]
        num_tracing_cats = len(tracing_cats_indices)
        
        if num_tracing_cats > 0:

            rand_factor = rng.random((num_tracing_cats, dim)) # Factor aleatorio para la actualización de velocidad
            V[tracing_cats_indices] += rand_factor * (G - X[tracing_cats_indices])
            np.clip(V[tracing_cats_indices], -velocity_limit, velocity_limit, out=V[tracing_cats_indices])  # Limita las velocidades
            X[tracing_cats_indices] += V[tracing_cats_indices]  # Actualiza las posiciones
            clamp_vec_np(X[tracing_cats_indices], bounds, in_place=True)  # Asegura que las posiciones estén dentro de los límites
            F[tracing_cats_indices] = objective(X[tracing_cats_indices])  # Evalúa el fitness de estos gatos


        # --- Actualizar Mejor Global ---
        current_best_idx = np.argmin(F)
        if F[current_best_idx] < Gf:
            Gf = F[current_best_idx]
            G = X[current_best_idx].copy()

        # --- Guardando en historial ---
        hist["best_f"].append(Gf)
        hist["mean_f"].append(np.mean(F))
        hist["best_x"].append(G.copy())
        hist["gbest"].append(G.copy())
        if log_positions and dim == 2 and (it % log_every == 0):
            hist["pos"].append(X.copy()) # Guarda las posiciones actuales


    return G, Gf, hist