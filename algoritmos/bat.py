from __future__ import annotations
import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# Algoritmo de Murciélago (BA) - NumPy
# ==================================
def bat(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        A: float = 0.9,     # Loudness (Sonoridad) inicial
        r: float = 0.9,     # Pulse rate (Tasa de pulso) inicial
        Qmin: float = 0.0,  # Frecuencia mínima
        Qmax: float = 2.0,  # Frecuencia máxima
        alpha: float = 0.9, # Factor de reducción de Loudness
        gamma: float = 0.9, # Factor de incremento de Pulse rate
        sigma: float = 0.01,# Desviación para el paseo aleatorio
        log_positions: bool = False, log_every: int = 1):
    """
    Implementación del Algoritmo de Murciélago (Bat Algorithm) usando NumPy.
    Basado en https://pypi.org/project/BatAlgorithm/#description.
    """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Parámetros del algoritmo
    A_vec = np.full(pop_size, A) # Vector de Sonoridad (Loudness)
    R_vec = np.full(pop_size, r) # Vector de Tasa de Pulso (Pulse Rate)

    # Inicializa las posiciones de los murciélagos
    X = rand_vec_in_bounds_np(bounds, pop_size, rng)
    # Inicializa las velocidades
    V = np.zeros((pop_size, dim))
    
    # Evalúa la población inicial
    F = objective(X)
    
    # Encuentra el mejor global inicial
    g_idx = np.argmin(F)
    G = X[g_idx].copy()
    Gf = F[g_idx]

    # Prepara el historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = []

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        
        # --- Generar nuevas soluciones (movimiento global) ---
        
        # Genera frecuencias aleatorias (beta)
        Q = Qmin + (Qmax - Qmin) * rng.random((pop_size, 1))
        
        # Ecuación de velocidad 
        V = V + (X - G) * Q
        
        # Ecuación de posición
        X_new = X + V
        
        # --- Búsqueda local (paseo aleatorio) ---
        
        # Identifica qué murciélagos harán búsqueda local (si rnd > r)
        local_search_mask = rng.random(pop_size) > R_vec
        num_local = np.sum(local_search_mask)
        
        if num_local > 0:
            # Genera un paseo aleatorio alrededor de la *mejor* solución (G)
            # El 'sigma' controla qué tan lejos es el paseo
            avg_F = np.mean(F) # Promedio de fitness actual
            epsilon = sigma * avg_F if avg_F != 0 else sigma # Promedio de sonoridad
            
            walk = G + epsilon * rng.standard_normal(size=(num_local, dim))
            
            # Reemplaza las soluciones de X_new con el paseo aleatorio
            X_new[local_search_mask] = walk

        # Limita todas las nuevas soluciones a los bounds
        clamp_vec_np(X_new, bounds, in_place=True)
        
        # Evalúa las nuevas soluciones
        F_new = objective(X_new)

        # --- Aceptación de soluciones (Loudness y Pulso) ---
        
        # Máscara de aceptación:
        # 1. La nueva solución es mejor (F_new < F)
        # 2. Y un número aleatorio es menor que la Sonoridad (A_vec)
        accept_mask = (F_new < F) & (rng.random(pop_size) < A_vec)
        
        # Actualiza las posiciones y fitness de los murciélagos aceptados
        X[accept_mask] = X_new[accept_mask]
        F[accept_mask] = F_new[accept_mask]
        
        # Actualiza los parámetros A y r para los murciélagos aceptados
        # A (Loudness) disminuye
        A_vec[accept_mask] *= alpha
        # R (Pulse rate) aumenta (se acerca a r_inicial)
        R_vec[accept_mask] = r * (1 - np.exp(-gamma * (it + 1)))

        # --- Actualizar el mejor global ---
        # Revisa si alguna de las *nuevas* posiciones aceptadas es el
        # nuevo mejor global.
        g_idx = np.argmin(F)
        if F[g_idx] < Gf:
            Gf = F[g_idx]
            G = X[g_idx].copy()

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