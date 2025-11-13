# algoritmos/aco.py
from __future__ import annotations
import numpy as np
from typing import Callable

# Importamos desde el 'utils.py' que está en la carpeta raíz
from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# ==================================
# ACO para Dominios Continuos (ACOR)
# ==================================
def aco(objective: Callable, bounds: np.ndarray,
        max_iters: int, pop_size: int, seed: int | None = None,
        k: int | None = None,
        q: float = 0.5,
        xi: float = 0.85,
        log_positions: bool = False, log_every: int = 1):
    """
    Optimización por Colonia de Hormigas para Dominios Continuos (ACOR).
    
    Parámetros:
    - pop_size (n_ants): Número de hormigas (nuevas soluciones) a generar por iteración.
    - k (archive_size): Tamaño del archivo de soluciones (la "feromona").
    - q: Parámetro de "codicia" para la selección. Un 'q' bajo da más
         probabilidad a las mejores soluciones.
    - xi: Parámetro de "evaporación" o velocidad de convergencia.
    """
    
    # 1. Inicialización
    # -------------------
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    
    # Si k (tamaño de archivo) no se especifica, lo igualamos a pop_size
    archive_size = k if k is not None and k > 0 else pop_size
    
    # Inicializa el Archivo de Soluciones (nuestra "feromona")
    # Genera 'k' soluciones aleatorias
    archive_solutions = rand_vec_in_bounds_np(bounds, archive_size, rng)
    # Evalúa el archivo
    archive_fitness = objective(archive_solutions)
    
    # Ordena el archivo por fitness (el mejor primero)
    sort_indices = np.argsort(archive_fitness)
    archive_solutions = archive_solutions[sort_indices]
    archive_fitness = archive_fitness[sort_indices]
    
    # Guarda el mejor global
    best_x = archive_solutions[0].copy()
    best_f = archive_fitness[0]

    # Prepara el historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = [] # 'pos' guardará las *nuevas* soluciones (hormigas)

    # 2. Bucle principal (Iteraciones)
    # ---------------------------------
    for it in range(max_iters):
        
        # --- Calcular Pesos (Feromona) ---
        # Asigna un "peso" a cada solución en el archivo basado en su rango (fitness).
        # Esto usa una función Gaussiana. Las mejores (rango bajo) tienen más peso.
        ranks = np.arange(archive_size)
        weights = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * \
                  np.exp(-((ranks) ** 2) / (2 * q**2 * archive_size**2))
        
        # Convierte los pesos en probabilidades (deben sumar 1)
        probabilities = weights / np.sum(weights)

        # --- Selección Probabilística (Hormigas eligen) ---
        # Cada una de las 'pop_size' hormigas elige una solución "guía" del
        # archivo, basándose en las probabilidades.
        chosen_indices = rng.choice(archive_size, size=pop_size, p=probabilities)
        chosen_solutions = archive_solutions[chosen_indices]
        
        # --- Muestreo Gaussiano (Generar nuevas soluciones) ---
        # Cada hormiga genera una nueva solución "olfateando" alrededor de
        # la solución guía que eligió.
        
        # Calculamos la desviación estándar (sigma) para cada dimensión
        # basado en la desviación de TODO el archivo. Esto promueve la
        # exploración donde el archivo está más "disperso".
        sigmas = np.std(archive_solutions, axis=0) * xi
        
        # Genera 'pop_size' nuevas soluciones
        # 'loc' es el centro (las soluciones guía elegidas)
        # 'scale' es la desviación (las sigmas que calculamos)
        new_solutions = rng.normal(loc=chosen_solutions, scale=sigmas, size=(pop_size, dim))
        
        # Limita las nuevas soluciones a los 'bounds'
        clamp_vec_np(new_solutions, bounds, in_place=True)
        
        # Evalúa a las nuevas hormigas
        new_fitness = objective(new_solutions)

        # --- Actualización del Archivo (Evaporación y Depósito) ---
        # Juntamos el archivo viejo y las nuevas soluciones en una "piscina"
        pool_solutions = np.vstack([archive_solutions, new_solutions])
        pool_fitness = np.concatenate([archive_fitness, new_fitness])
        
        # Ordenamos la piscina
        sort_indices = np.argsort(pool_fitness)
        
        # Los 'k' mejores sobreviven para ser el nuevo archivo
        archive_solutions = pool_solutions[sort_indices[:archive_size]]
        archive_fitness = pool_fitness[sort_indices[:archive_size]]
        
        # Actualiza el mejor global
        if archive_fitness[0] < best_f:
            best_f = archive_fitness[0]
            best_x = archive_solutions[0].copy()

        # --- Guardando en historial ---
        hist["best_f"].append(best_f)
        hist["mean_f"].append(np.mean(new_fitness)) # Fitness promedio de las *nuevas* hormigas
        hist["best_x"].append(best_x.copy())
        hist["gbest"].append(best_x.copy())
        if log_positions and dim == 2 and (it % log_every == 0):
            hist["pos"].append(new_solutions.copy()) # Guarda las posiciones de las hormigas

    return best_x, best_f, hist