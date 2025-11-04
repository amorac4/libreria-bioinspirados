from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Algoritmo de Optimización por Enjambre de Gatos (CSO) - NumPy
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
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Inicializa las posiciones de los gatos (soluciones)
    # X (pop_size, dim)
    X = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # Inicializa las velocidades de los gatos (matrices)
    # V (pop_size, dim)
    V = np.zeros((pop_size, dim))
    
    # Evalúa la población inicial
    F = objective(X) # Fitness de cada gato
    
    # Encuentra el mejor global inicial
    g_idx = np.argmin(F)
    G = X[g_idx].copy() # Mejor posición global
    Gf = F[g_idx]       # Mejor fitness global

    # Prepara el historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
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
        
        # Selecciona los gatos en modo de búsqueda
        seeking_cats_indices = np.where(is_seeking_mode)[0]
        num_seeking_cats = len(seeking_cats_indices)
        
        if num_seeking_cats > 0:
            current_seeking_positions = X[seeking_cats_indices] # Posiciones actuales de estos gatos
            
            # Genera múltiples candidatos para cada gato en modo de búsqueda
            # (num_seeking_cats, seeking_memory_pool, dim)
            # Cada candidato se genera alrededor de la posición actual del gato
            # El rango es un factor del rango total de los límites
            seeking_range = bounds_range * seeking_range_factor
            
            # Genera desplazamientos aleatorios para todos los candidatos
            # uniformemente entre -seeking_range y +seeking_range
            displacements = rng.uniform(-1, 1, (num_seeking_cats, seeking_memory_pool, dim)) * seeking_range
            
            # Suma los desplazamientos a las posiciones actuales para obtener los candidatos
            # current_seeking_positions[:, np.newaxis, :] expande para que se pueda sumar
            candidate_positions = current_seeking_positions[:, np.newaxis, :] + displacements
            
            # Asegura que los candidatos estén dentro de los límites
            for j in range(num_seeking_cats):
                clamp_vec_np(candidate_positions[j], bounds, in_place=True)
            
            # Evalúa todos los candidatos
            # Reshape para evaluar todos a la vez (num_seeking_cats * seeking_memory_pool, dim)
            flat_candidates = candidate_positions.reshape(-1, dim)
            flat_fitness = objective(flat_candidates)
            
            # Reshape de nuevo a (num_seeking_cats, seeking_memory_pool) para encontrar el mejor por gato
            reshaped_fitness = flat_fitness.reshape(num_seeking_cats, seeking_memory_pool)
            
            # Encuentra el mejor candidato para cada gato
            best_candidate_indices_per_cat = np.argmin(reshaped_fitness, axis=1)
            
            # Actualiza las posiciones de los gatos en modo de búsqueda con su mejor candidato
            # Esto se hace seleccionando la fila correspondiente de cada candidato_positions
            X[seeking_cats_indices] = candidate_positions[
                np.arange(num_seeking_cats), best_candidate_indices_per_cat
            ]
            F[seeking_cats_indices] = reshaped_fitness[
                np.arange(num_seeking_cats), best_candidate_indices_per_cat
            ]
            # Las velocidades se resetean o se ignoran en modo de búsqueda,
            # para simplificar las dejamos tal cual o se pueden resetear a 0.
            # En esta implementación, las velocidades se modifican solo en modo de rastreo.


        # --- MODO DE RASTREO (Tracing Mode) ---
        # Los gatos "cazan" y se mueven hacia el mejor global
        
        # Selecciona los gatos en modo de rastreo
        tracing_cats_indices = np.where(~is_seeking_mode)[0]
        num_tracing_cats = len(tracing_cats_indices)
        
        if num_tracing_cats > 0:
            # Actualiza las velocidades
            # V[tracing_cats_indices] es una vista de las velocidades de estos gatos
            # G - X[tracing_cats_indices] es la dirección hacia el mejor global
            
            # Un número aleatorio diferente para cada gato en cada dimensión
            rand_factor = rng.random((num_tracing_cats, dim))
            
            V[tracing_cats_indices] += rand_factor * (G - X[tracing_cats_indices])
            
            # Limita las velocidades
            # velocity_limit es un array (dim,)
            np.clip(V[tracing_cats_indices], -velocity_limit, velocity_limit, out=V[tracing_cats_indices])
            
            # Actualiza las posiciones
            X[tracing_cats_indices] += V[tracing_cats_indices]
            
            # Asegura que las nuevas posiciones estén dentro de los límites
            clamp_vec_np(X[tracing_cats_indices], bounds, in_place=True)
            
            # Evalúa las nuevas posiciones de los gatos en modo de rastreo
            F[tracing_cats_indices] = objective(X[tracing_cats_indices])


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

    # 3. Fin
    # --------
    return G, Gf, hist