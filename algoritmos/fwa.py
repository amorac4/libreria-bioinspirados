from __future__ import annotations
import numpy as np
from typing import Callable
from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Fireworks Algorithm (FWA) - NumPy
# Basado en el paper de Tan y Zhu (2010)
# ==================================
def fwa(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        S_min: int = 5,       # Número mínimo de chispas por explosión
        S_max: int = 30,      # Número máximo de chispas por explosión
        A_max: float = 40.0,  # Amplitud máxima de explosión
        m_gauss: int = 5,     # Número de chispas de "mutación" Gaussiana
        log_positions: bool = False, log_every: int = 1):

    
    # 1. Inicialización
    # ------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Inicializa las 'n' (pop_size) posiciones de los fuegos artificiales
    X = rand_vec_in_bounds_np(bounds, pop_size, rng)
    
    # *** OPTIMIZACIÓN ***
    # Evalúa la población inicial
    F = objective(X)
    
    # Guarda el mejor inicial
    best_idx = np.argmin(F)
    G = X[best_idx].copy()  # Global best position
    Gf = F[best_idx]       # Global best fitness

    # Prepara el historial
    hist_keys = ["best_f", "mean_f", "best_x", "gbest"]
    if log_positions:
        hist_keys.extend(["pos", "gbest_hist"])
    hist = init_history(keys=hist_keys)

    # Constante para evitar divisiones por cero
    EPS = 1e-9

    # 2. Ciclo principal de FWA
    # -------------------------
    for it in range(max_iters):
        
        # --- Cálculo de Amplitudes y Chispas (Vectorizado) ---
        
        f_min = np.min(F)
        f_max = np.max(F)
        f_range_inv = 1.0 / (f_max - f_min + EPS)
        
        # Sumas para normalización
        f_sum_norm_s = np.sum((f_max - F) * f_range_inv) + EPS
        f_sum_norm_a = np.sum((F - f_min) * f_range_inv) + EPS
        
        # S: Número de chispas (vector)
        # (f_max - F) -> Buen fitness (bajo F) da un valor alto
        S_norm = (f_max - F) * f_range_inv
        S = S_min + (S_max - S_min) * (S_norm / f_sum_norm_s)
        S = np.clip(np.round(S), S_min, S_max).astype(int)

        # A: Amplitud de explosión (vector)
        # (F - f_min) -> Buen fitness (bajo F) da un valor bajo
        A_norm = (F - f_min) * f_range_inv
        A = A_max * (A_norm / f_sum_norm_a)

        # --- Generación de Chispas ---
        
        # Pool para guardar todas las nuevas chispas y sus fitness
        all_sparks_X = []
        all_sparks_F = []

        # Fase 1: Chispas de Explosión (basado en Algoritmo 1 del paper)
        for i in range(pop_size):
            n_sparks_i = S[i]
            A_i = A[i]
            
            # Si no hay chispas que generar, saltamos
            if n_sparks_i == 0:
                continue

            # Vector de chispas (copias del padre)
            new_sparks = np.tile(X[i], (n_sparks_i, 1))
            
            # Selecciona 'z' dimensiones al azar para modificar
            # El paper sugiere z = round(dim * rand(0,1))
            z = rng.integers(1, dim + 1)
            dims_to_change = rng.choice(dim, z, replace=False)
            
            # Genera desplazamientos aleatorios
            # h = A_i * rand(-1, 1)
            displacement = A_i * rng.uniform(-1, 1, (n_sparks_i, z))
            
            # Aplica el desplazamiento SÓLO a las dimensiones elegidas
            new_sparks[:, dims_to_change] += displacement
            
            # Restringe a los límites
            clamp_vec_np(new_sparks, bounds, in_place=True)
            
            # Evalúa y guarda
            all_sparks_X.append(new_sparks)
            all_sparks_F.append(objective(new_sparks))

        # Fase 2: Chispas de Mutación Gaussiana (basado en Algoritmo 2)
        # Para mantener la diversidad
        gauss_sparks_X = []
        for _ in range(m_gauss):
            # Elige un fuego artificial base al azar
            i = rng.integers(pop_size)
            
            new_spark = X[i].copy()
            
            # Elige 'z' dimensiones al azar
            z = rng.integers(1, dim + 1)
            dims_to_change = rng.choice(dim, z, replace=False)

            # Aplica mutación Gaussiana: x_k = x_k * N(1, 1)
            scale = rng.normal(1.0, 1.0, z)
            new_spark[dims_to_change] *= scale
            
            # Restringe a los límites
            clamp_vec_np(new_spark, bounds, in_place=True)
            
            gauss_sparks_X.append(new_spark)

        if m_gauss > 0:
            gauss_sparks_X = np.array(gauss_sparks_X)
            all_sparks_X.append(gauss_sparks_X)
            all_sparks_F.append(objective(gauss_sparks_X))

        # --- Selección de la Siguiente Generación ---
        
        # Junta los fuegos artificiales padres, las chispas de explosión y las chispas gaussianas
        pool_X = np.vstack([X] + all_sparks_X)
        pool_F = np.concatenate([F] + all_sparks_F)
        
        # Ordena el pool completo por fitness
        sorted_indices = np.argsort(pool_F)
        
        # Elitismo: Los 'pop_size' mejores sobreviven para ser la siguiente
        # generación de fuegos artificiales
        X = pool_X[sorted_indices[:pop_size]]
        F = pool_F[sorted_indices[:pop_size]]
        
        # Actualiza el mejor global (G)
        # El mejor actual siempre estará en la posición 0 del nuevo X/F
        if F[0] < Gf:
            Gf = F[0]
            G = X[0].copy()

        # --- Guardando en historial ---
        hist["best_f"].append(Gf)
        hist["mean_f"].append(np.mean(F))
        hist["best_x"].append(G.copy())
        hist["gbest"].append(G.copy())
        if log_positions and (it % log_every == 0):
            hist["pos"].append(X.copy())
            hist["gbest_hist"].append(G.copy())

    # 3. Retorno
    # -----------
    return G, Gf, hist