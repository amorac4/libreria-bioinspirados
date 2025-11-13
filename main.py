from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Callable, Any
import numpy as np

from algoritmos import ALGORITMOS_REGISTRADOS
from objetivos import OBJETIVOS_REGISTRADOS
from visualizar import  animate_convergencia, animate_swarm, animate_swarm_heatmap, plot_convergencia, animate_swarm_3d

DEFAULT_CONFIG_PATHS = ["config.local.json", "config.json"]

def load_config() -> dict:
    """Busca config.local.json o config.json en el directorio actual."""
    for name in DEFAULT_CONFIG_PATHS:
        p = Path(name)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"Error leyendo {name}: {e}")
    # Si no hay config, devolvemos defaults mínimos
    return {
        "run": {
            "alg": "pso",
            "obj": "ackley",
            "dim": 2,
            "bounds": "-5,5",
            "iters": 200,
            "pop": 50,
            "seed": 7,
            "plot": False,
            "animate": True
        },
        "pso": { "w": 0.72, "c1": 1.49, "c2": 1.49, "vmax": None },
        "ga":  { "pc": 0.9, "pm": 0.05, "elitism": 1, "tournament_k": 3, "blx_alpha": 0.3, "sigma_frac": 0.1 },
        "de":  { "F": 0.5, "CR": 0.9, "strategy": "rand/1/bin" }
    }

def parse_bounds(bounds_spec: Any, dim: int) -> np.ndarray:
    """ Parsea la especificación de límites y devuelve un array de NumPy (dim, 2). """
    pairs = []
    if isinstance(bounds_spec, str):
        s = bounds_spec.strip()
        if ";" in s:
            # Caso: "-5,5;-1,1"
            for part in s.split(";"):
                lo, hi = map(float, part.split(","))
                pairs.append((lo, hi))
            if len(pairs) != dim:
                raise ValueError(f"Los límites ({len(pairs)}) no coinciden con dim={dim}.")
        else:
            # Caso: "-5,5"
            lo, hi = map(float, s.split(","))
            pairs = [(lo, hi)] * dim

    elif isinstance(bounds_spec, (list, tuple)):
        # Caso: [[-5,5],[-1,1],...]
        pairs = [(float(lo), float(hi)) for lo, hi in bounds_spec]
        if len(pairs) == 1:
            pairs = pairs * dim
        if len(pairs) != dim:
            raise ValueError(f"Los límites ({len(pairs)}) no coinciden con dim={dim}.")
    else:
        raise TypeError("Formato de 'bounds' no reconocido.")

    return np.array(pairs)

# main.py

def main():
    cfg = load_config()
    
    # Carga la configuración de la corrida (UNA SOLA)
    run = cfg.get("run", {})
    if not run:
        print("[ERROR] No se encontró la sección 'run' en el config.json.")
        sys.exit(1)

    # --- 1. Cargar Parámetros Comunes (Fuera del Bucle) ---
    print("Cargando configuración común...")
    
    # Parámetros del algoritmo
    alg_name = run.get("alg")
    alg_fn = ALGORITMOS_REGISTRADOS.get(alg_name)
    if not alg_fn:
        print(f"[ERROR] Algoritmo '{alg_name}' no encontrado en ALGORITMOS_REGISTRADOS.")
        sys.exit(1)
    alg_params = cfg.get(alg_name, {})

    # Parámetros de la simulación
    dim = run.get("dim", 2)
    iters = run.get("iters", 200)
    pop = run.get("pop", 50)
    seed = run.get("seed", None)

    # Parámetros de visualización
    plot = run.get("plot", False)
    animate = run.get("animate", True)
    show_heatmap = run.get("show_heatmap", True)
    animation_type = run.get("animation_type", "2d")
    heatmap_res = run.get("heatmap_res", 100)
    heatmap_levels = run.get("heatmap_levels", 20)
    cmap = run.get("cmap", "viridis")
    save_path = run.get("save_path", None)

    # Procesar Límites (Bounds)
    bounds_str = run.get("bounds", "-5,5")
    try:
        b_min, b_max = [float(b.strip()) for b in bounds_str.split(',')]
        bounds = np.array([[b_min, b_max]] * dim)
        print(f"Límites comunes: [{b_min}, {b_max}] en {dim}D")
    except Exception as e:
        print(f"[ERROR] 'bounds' debe ser 'min,max' (ej: '-5,5'). Error: {e}")
        sys.exit(1)

    # --- 2. Preparar Bucle de Funciones Objetivo ---
    
    obj_name_string = run.get("obj")
    if not obj_name_string:
        print("[ERROR] No se especificó 'obj' en la sección 'run'.")
        sys.exit(1)

    # Crea la lista de objetivos (ej: "sphere, rastrigin" -> ["sphere", "rastrigin"])
    obj_names_list = [name.strip() for name in obj_name_string.split(',') if name.strip()]
    n_exp = len(obj_names_list)
    print(f"Configuración cargada. {n_exp} funciones objetivo en cola.")

    # --- 3. Iniciar Bucle de Ejecución ---
    
    for i, obj_name in enumerate(obj_names_list):
        
        print(f"\n[--- Ejecución {i+1}/{n_exp}: Alg={alg_name}, Obj={obj_name} ---]")

        # Cargar la función objetivo (DENTRO del bucle)
        objective = OBJETIVOS_REGISTRADOS.get(obj_name)
        if not objective:
            print(f"[ERROR] Función objetivo '{obj_name}' no encontrada. Omitiendo.")
            continue # Salta a la siguiente función

        # Copia los parámetros para esta corrida (evita contaminación)
        alg_params_copy = alg_params.copy()
        
        # Activa el log de posiciones si es necesario (DENTRO del bucle)
        if animate and dim == 2:
            alg_params_copy.update({"log_positions": True, "log_every": 1})
        
        # Modifica el seed para que cada corrida sea única (opcional pero recomendado)
        current_seed = seed + i if seed is not None else None
        
        print(f"Ejecutando (dim={dim}, iters={iters}, pop={pop}, seed={current_seed})...")

        # --- Ejecutar el algoritmo ---
        best_x, best_f, history = alg_fn(
            objective=objective,
            bounds=bounds,
            max_iters=iters,
            pop_size=pop,
            seed=current_seed, 
            **alg_params_copy
        )

        print(f"[OK] Resultado para {obj_name}:")
        print(f" best_f = {best_f:.6f}")
        print(f" best_x = {best_x}")

        # --- Gráficas (DENTRO del bucle) ---
        if plot:
            plot_convergencia(history, title=f"Convergencia {alg_name.upper()} - {obj_name}")

        if animate:
            if dim == 2 and "pos" in history:
                
                # --- Función auxiliar para 2D ---
                def run_2d_animation():
                    if show_heatmap:
                        print("Mostrando animación 2D (Heatmap)...")
                        animate_swarm_heatmap(history, objective, bounds,
                                            title=f"{alg_name.upper()} - {obj_name} (2D Heatmap)",
                                            res=heatmap_res, levels=heatmap_levels, cmap=cmap,
                                            save_path=save_path)
                    else:
                        print("Mostrando animación 2D (Scatter)...")
                        animate_swarm(history, bounds,
                                    title=f"{alg_name.upper()} - {obj_name} (2D Scatter)",
                                    save_path=save_path)

                # --- Función auxiliar para 3D ---
                def run_3d_animation():
                    print("Mostrando animación 3D... (cierre la ventana para continuar)")
                    animate_swarm_3d(history, objective, bounds,
                                        title=f"{alg_name.upper()} - {obj_name} (3D Surface)",
                                        res=heatmap_res, cmap=cmap,
                                        save_path=save_path)
                
                # --- Lógica de Selección ---
                if animation_type == "3d":
                    run_3d_animation()
                elif animation_type == "2d":
                    run_2d_animation()
                elif animation_type == "both":
                    run_3d_animation()  # Muestra 3D primero
                    run_2d_animation()  # Muestra 2D después
                else:
                    print(f"Advertencia: 'animation_type' ('{animation_type}') no reconocido. Usando '2d'.")
                    run_2d_animation()

            else: # Si dim no es 2, o no hay historial de 'pos'
                animate_convergencia(history,
                                     title=f"Convergencia {alg_name.upper()} - {obj_name}",
                                     save_path=save_path)

    print("\n[--- Fin de todas las ejecuciones ---]")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)