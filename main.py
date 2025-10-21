from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Callable, Any
import numpy as np

from algoritmos import ALGORITMOS_REGISTRADOS
from objetivos import OBJETIVOS_REGISTRADOS
from visualizar import  animate_convergencia, animate_swarm, animate_swarm_heatmap, plot_convergencia

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

def main():
    cfg = load_config()
    run = cfg.get("run", {})

    alg_name: str = run.get("alg", "pso")
    obj_name: str = run.get("obj", "ackley")
    dim: int = int(run.get("dim", 2))
    bounds_spec = run.get("bounds", "-5,5")
    iters: int = int(run.get("iters", 200))
    pop: int = int(run.get("pop", 50))
    seed = run.get("seed", None)
    plot: bool = bool(run.get("plot", False))
    animate: bool = bool(run.get("animate", False))
    show_heatmap: bool = bool(run.get("show_heatmap", False))
    heatmap_res: int = int(run.get("heatmap_res", 200))
    heatmap_levels: int = int(run.get("heatmap_levels", 30))
    cmap: str = run.get("cmap", "viridis")
    save_path = run.get("save_path", None)


    if alg_name not in ALGORITMOS_REGISTRADOS:
        raise ValueError(f"Algoritmo desconocido '{alg_name}'. Opciones: {list(ALGORITMOS_REGISTRADOS.keys())}")
    if obj_name not in OBJETIVOS_REGISTRADOS:
        raise ValueError(f"Objetivo desconocido '{obj_name}'. Opciones: {list(OBJETIVOS_REGISTRADOS.keys())}")
    if dim <= 0:
        raise ValueError("dim debe ser > 0")
    if iters <= 0 or pop <= 0:
        raise ValueError("iters y pop deben ser > 0")

    bounds = parse_bounds(bounds_spec, dim)
    objective: Callable = OBJETIVOS_REGISTRADOS[obj_name]
    alg_fn = ALGORITMOS_REGISTRADOS[alg_name]

    # hiperparámetros del algoritmo desde el tope del config (p.ej. cfg["pso"])
    alg_params = cfg.get(alg_name, {}) if isinstance(cfg.get(alg_name), dict) else {}

    # habilitar registro de posiciones si se pide animación 2D
    if animate and dim == 2:
        alg_params = {**alg_params, "log_positions": True, "log_every": 1}

    best_x, best_f, history = alg_fn(
        objective=objective,
        bounds=bounds,
        max_iters=iters,
        pop_size=pop,
        seed=seed,
        **alg_params
    )

    print(f"[OK] Algoritmo: {alg_name} | Objetivo: {obj_name}")
    print(f" best_f = {best_f:.6f}")
    print(f" best_x = {best_x}")

    if plot:
        plot_convergencia(history, title=f"Convergencia {alg_name.upper()} - {obj_name}")

    if animate:
        if dim == 2 and "pos" in history:
            if show_heatmap:
                animate_swarm_heatmap(history, objective, bounds,
                                    title=f"{alg_name.upper()} - {obj_name}",
                                    res=heatmap_res, levels=heatmap_levels, cmap=cmap,
                                    save_path=save_path)
            else:
                animate_swarm(history, bounds,
                            title=f"{alg_name.upper()} - {obj_name}",
                            save_path=save_path)
        else:
            animate_convergencia(history,
                                title=f"{alg_name.upper()} - {obj_name}",
                                save_path=save_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)