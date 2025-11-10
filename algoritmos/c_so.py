from __future__ import annotations
import numpy as np
from typing import Callable
from math import gamma

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Algoritmo de cukoo swarm optimization (CSO)
# ==================================

def _get_levy_step(rng: np.random.Generator, beta: float, size: tuple[int, ...]) -> np.ndarray:
    """ Genera pasos de Lévy vectorizados """
    
    # Calcular sigmas 
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
    sigma_u = (num / den)**(1 / beta)
    sigma_v = 1
    
    # Generar pasos u y v
    u = rng.normal(0, sigma_u, size)
    v = rng.normal(0, sigma_v, size)
    
    # Calcular el paso S
    S = u / (np.abs(v)**(1 / beta))
    return S

def c_so(objective: Callable, bounds: np.ndarray,
       max_iters: int, pop_size: int, seed: int | None = None,
       pa: float = 0.25,                 # Probabilidad de abandono de nido
       beta: float = 1.5,                # Parámetro de vuelo de Lévy
       alpha: float = 0.01,              # Factor de escala del paso de Lévy
       log_positions: bool = False, log_every: int = 1):
   
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed) 
    dim = bounds.shape[0] 
    
    # Inicializar 'nidos' (población)
    X = rand_vec_in_bounds_np(bounds, pop_size, rng) #
    F = objective(X) # Fitness de cada nido
    
    # Encontrar el mejor nido inicial
    g_idx = np.argmin(F) 
    G = X[g_idx].copy() # Mejor posición global (mejor nido)
    Gf = F[g_idx]       # Mejor fitness global
    
    # Preparar historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest")) #
    if log_positions and dim == 2:
        hist["pos"] = []
    
    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        
        # --- Parte 1: Vuelos de Lévy (update_position_1) ---
        # Generar nuevos cucos (soluciones) mediante vuelos de Lévy
        
        # Obtener pasos de Lévy vectorizados
        levy_steps = _get_levy_step(rng, beta, size=(pop_size, dim)) #
        
        # Calcular el tamaño del paso (diferente para cada nido)
        # En CS, el paso se escala con la diferencia a la *mejor* solución actual (G)
        step_size = alpha * levy_steps * (X - G) #
        
        # Generar nuevas posiciones y evaluarlas
        X_new = X + rng.normal(size=(pop_size, dim)) * step_size 
        clamp_vec_np(X_new, bounds, in_place=True) #
        F_new = objective(X_new)
        
        # Selección (Greedy): quedarse con el mejor entre el viejo y el nuevo
        # Compara el fitness del nido 'i' con el del nuevo cuco 'i'
        improvement_mask = F_new < F
        X[improvement_mask] = X_new[improvement_mask]
        F[improvement_mask] = F_new[improvement_mask]

        # --- Parte 2: Abandono de nidos (update_position_2) ---
        # Reemplazar una fracción 'pa' de los peores nidos
        
        # Identificar qué nidos se abandonan
        abandon_mask = rng.random(pop_size) < pa
        abandon_indices = np.where(abandon_mask)[0]
        num_abandon = len(abandon_indices)
        
        if num_abandon > 0:
            # Generar nuevas soluciones para los nidos abandonados
            # Esto se hace con pasos aleatorios basados en la diferencia de otros dos nidos
            
            # Elige dos nidos aleatorios (d1, d2) por cada nido a abandonar
            d1_indices = rng.choice(pop_size, num_abandon, replace=True)
            d2_indices = rng.choice(pop_size, num_abandon, replace=True)
            
            # Genera el nuevo paso vectorizado
            X_d1 = X[d1_indices]
            X_d2 = X[d2_indices]
            step_size_abandon = rng.random((num_abandon, dim)) * (X_d1 - X_d2) #
            
            # Calcula las nuevas posiciones y su fitness
            X_new_abandon = X[abandon_indices] + step_size_abandon
            clamp_vec_np(X_new_abandon, bounds, in_place=True) #
            F_new_abandon = objective(X_new_abandon)
            
            # Selección (Greedy): Reemplaza el nido abandonado (X[i])
            # solo si la nueva solución (X_new_abandon) es mejor.
            
            # Fitness actual de los nidos que van a ser abandonados
            current_fitness_abandon = F[abandon_indices] 
            
            # Máscara de mejora (local a los 'num_abandon' nidos)
            improvement_mask_abandon = F_new_abandon < current_fitness_abandon
            
            # Índices locales (0 a num_abandon-1) que mejoraron
            update_indices_local = np.where(improvement_mask_abandon)[0]
            
            if update_indices_local.size > 0:
                # Índices globales (en X y F) de los nidos a actualizar
                update_indices_global = abandon_indices[update_indices_local]
                
                # Aplicar las actualizaciones SÓLO a los nidos que mejoraron
                X[update_indices_global] = X_new_abandon[update_indices_local]
                F[update_indices_global] = F_new_abandon[update_indices_local]


        # --- Actualizar Mejor Global ---
        # Después de ambas fases, encontrar el mejor nido actual
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
            hist["pos"].append(X.copy()) 


    return G, Gf, hist