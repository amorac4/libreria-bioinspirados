# algoritmos/cso.py
from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Algoritmo de Optimización por Enjambre de Gatos (CSO) - (Según el Paper)
# ==================================
def cso(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        mix_rate: float = 0.5,           # MR: Proporción de gatos en modo búsqueda
        smp: int = 5,                    # SMP: Seeking Memory Pool (Nº de candidatos)
        srd: float = 0.2,                # SRD: Seeking Range of Dimension (Factor de rango)
        cdc: float = 0.8,                # CDC: Counts of Dimensions to Change (Probabilidad de mutar dim)
        spc: bool = True,                # SPC: Self Position Considering (Considerar la pos. actual)
        velocity_limit_factor: float = 0.03, # VL: Límite de velocidad
        log_positions: bool = False, log_every: int = 1):
    
    """
    Implementación vectorizada de Cat Swarm Optimization (CSO)
    basada en el paper de Chu, Tsai y Pan (2006).
    """
    
    # 1. Inicialización
    # ------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Rango total de los límites (para SRD y VL)
    bounds_range = bounds[:, 1] - bounds[:, 0]
    
    # Límite de velocidad (Tracing)
    velocity_limit = bounds_range * velocity_limit_factor
    # Rango de perturbación (Seeking)
    seeking_range = bounds_range * srd
    
    # Población
    X = rand_vec_in_bounds_np(bounds, pop_size, rng) # Posiciones
    V = np.zeros((pop_size, dim))                    # Velocidades
    F = objective(X)                                 # Fitness
    
    # Mejor Global
    g_idx = np.argmin(F)
    G = X[g_idx].copy()
    Gf = F[g_idx]

    # Historial
    hist_keys = ["best_f", "mean_f", "best_x", "gbest"]
    if log_positions:
        hist_keys.extend(["pos", "gbest_hist"])
    hist = init_history(keys=hist_keys)

    # 2. Ciclo Principal
    # ------------------
    for it in range(max_iters):
        
        # --- División de Modos (Determinista) ---
        # A diferencia de la versión anterior (estocástica), esta asegura
        # una división fija en cada iteración, lo cual es más estable.
        num_seeking = int(pop_size * mix_rate)
        num_tracing = pop_size - num_seeking
        
        # Mezcla los índices para elegir gatos al azar
        all_indices = rng.permutation(pop_size)
        seeking_indices = all_indices[:num_seeking]
        tracing_indices = all_indices[num_seeking:]

        # --- Modo de Búsqueda (Seeking Mode) ---
        if num_seeking > 0:
            # Implementación vectorizada del "Seeking Mode"
            
            # Cuántas copias nuevas generar (j)
            j = smp
            if spc:
                j = smp - 1 # Se reserva un lugar para la pos. actual
            
            if j > 0:
                # Obtiene los gatos "padre" (num_seeking, 1, dim)
                seeking_parents = X[seeking_indices][:, np.newaxis, :]
                
                # Crea j copias de cada padre (num_seeking, j, dim)
                candidate_copies = np.tile(seeking_parents, (1, j, 1))
                
                # --- Mutación (SRD y CDC) ---
                # Crea perturbaciones aleatorias
                perturbations = rng.uniform(-seeking_range, seeking_range, 
                                            size=(num_seeking, j, dim))
                
                # Crea máscara de CDC: qué dimensiones SÍ mutar
                # (Cada dimensión tiene una prob 'cdc' de ser mutada)
                cdc_mask = rng.random((num_seeking, j, dim)) < cdc
                
                # Anula las perturbaciones que no pasaron el filtro CDC
                perturbations[~cdc_mask] = 0.0
                
                # Aplica las mutaciones
                candidates_mutated = candidate_copies + perturbations
                clamp_vec_np(candidates_mutated, bounds, in_place=True)
                
                # Evalúa todos los candidatos (num_seeking * j)
                candidates_flat = candidates_mutated.reshape(-1, dim)
                candidates_fit_flat = objective(candidates_flat)
                # Re-organiza (num_seeking, j)
                candidates_fit = candidates_fit_flat.reshape(num_seeking, j)

            # --- Selección (SPC) ---
            parent_fit = F[seeking_indices] # Fitness de los padres (num_seeking,)

            if spc:
                # Si se considera la pos. actual (padre)
                if j > 0:
                    # Compara padres vs hijos
                    all_fits = np.hstack([parent_fit[:, np.newaxis], candidates_fit])
                    all_candidates = np.concatenate([seeking_parents, candidates_mutated], axis=1)
                else:
                    # Caso smp=1 y spc=True, solo se considera el padre
                    all_fits = parent_fit[:, np.newaxis]
                    all_candidates = seeking_parents
            else:
                # No se considera la pos. actual (solo los 'smp' hijos)
                all_fits = candidates_fit
                all_candidates = candidates_mutated

            # Encuentra el mejor candidato para cada gato (índice 0 a smp)
            best_indices = np.argmin(all_fits, axis=1) # (num_seeking,)
            
            # Obtiene el fitness y la posición del mejor candidato
            best_new_fits = all_fits[np.arange(num_seeking), best_indices]
            best_new_pos = all_candidates[np.arange(num_seeking), best_indices]
            
            # --- Actualización Greedy ---
            # Actualiza solo los gatos que encontraron una posición mejor
            update_mask = best_new_fits < parent_fit
            
            update_indices_global = seeking_indices[update_mask]
            X[update_indices_global] = best_new_pos[update_mask]
            F[update_indices_global] = best_new_fits[update_mask]

        # --- Modo de Rastreo (Tracing Mode) ---
        if num_tracing > 0:
            # Esta parte ya era vectorial y correcta.
            X_trace = X[tracing_indices]
            V_trace = V[tracing_indices]
            
            # Actualiza velocidad (hacia G)
            rand_factor = rng.random((num_tracing, dim))
            V_new = V_trace + rand_factor * (G - X_trace)
            np.clip(V_new, -velocity_limit, velocity_limit, out=V_new)
            
            # Actualiza posición
            X_new = X_trace + V_new
            clamp_vec_np(X_new, bounds, in_place=True)
            
            # Evalúa y actualiza
            F_new = objective(X_new)
            X[tracing_indices] = X_new
            V[tracing_indices] = V_new
            F[tracing_indices] = F_new

        # --- Actualizar Mejor Global ---
        # Revisa *toda* la población después de ambos modos
        current_best_idx = np.argmin(F)
        if F[current_best_idx] < Gf:
            Gf = F[current_best_idx]
            G = X[current_best_idx].copy()

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