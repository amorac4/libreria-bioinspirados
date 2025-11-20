# main.py
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

# --- IMPORTACIONES ---
from algoritmos import ALGORITMOS_REGISTRADOS
# Importamos también INFO_OBJETIVOS para saber los límites
from objetivos import OBJETIVOS_REGISTRADOS, INFO_OBJETIVOS 
from visualizar import (
    animate_convergencia, animate_swarm, animate_swarm_heatmap, animate_swarm_3d,
    plot_convergencia, plot_boxplot, plot_convergence_average, 
    plot_comparacion_algoritmos
)

DEFAULT_CONFIG_PATHS = ["config.local.json", "config.json"]

def load_config() -> dict:
    for name in DEFAULT_CONFIG_PATHS:
        p = Path(name)
        if p.exists():
            try: return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e: raise RuntimeError(f"Error leyendo {name}: {e}")
    return {}

def main():
    cfg = load_config()
    run = cfg.get("run", {})
    if not run:
        print("[ERROR] No se encontró 'run' en config.json"); sys.exit(1)

    # --- Configuración Base ---
    base_dim = run.get("dim", 2)
    base_iters = run.get("iters", 200)
    base_pop = run.get("pop", 50)
    base_seed = run.get("seed", None)
    n_runs = run.get("n_runs", 1)
    
    plot = run.get("plot", False)
    plot_individual = run.get("plot_individual", True) 
    animate = run.get("animate", True)
    show_heatmap = run.get("show_heatmap", True)
    animation_type = run.get("animation_type", "2d")
    
    heatmap_res = run.get("heatmap_res", 100)
    heatmap_levels = run.get("heatmap_levels", 20)
    cmap = run.get("cmap", "viridis")
    save_path = run.get("save_path", None)

    # --- Bounds Globales vs Auto ---
    # Si el usuario pone "auto", usaremos los del archivo objetivos.py
    config_bounds_str = run.get("bounds", "auto") 

    obj_str = run.get("obj", "sphere")
    obj_names = [x.strip() for x in obj_str.split(',') if x.strip()]

    alg_str = run.get("alg", "pso")
    alg_names = [x.strip() for x in alg_str.split(',') if x.strip()]

    print(f"=== ESTUDIO COMPARATIVO INTELIGENTE ===")
    print(f"Algoritmos: {alg_names}")
    print(f"Objetivos : {obj_names}")
    print(f"Config Global: Runs={n_runs}, Pop={base_pop}, Iters={base_iters}")
    print(f"Modo Bounds  : {config_bounds_str}\n")

    for i_obj, obj_name in enumerate(obj_names):
        objective = OBJETIVOS_REGISTRADOS.get(obj_name)
        if not objective:
            print(f"[WARN] Objetivo '{obj_name}' no encontrado."); continue
            
        # --- LÓGICA DE METADATOS (DIMENSIÓN Y BOUNDS) ---
        info = INFO_OBJETIVOS.get(obj_name, {})
        
        # 1. Determinar Dimensión
        # Si la función es FIJA en 2D (ej: himmelblau), forzamos dim=2
        if "fixed_dim" in info:
            current_dim = info["fixed_dim"]
            dim_msg = f"{current_dim} (Fijo)"
        else:
            # Si no es fija, usamos la del config
            current_dim = base_dim
            dim_msg = f"{current_dim} (Config)"
            
        # 2. Determinar Bounds
        if config_bounds_str == "auto":
            # Usar los bounds estándar de la función
            if "bounds" in info:
                b_val = info["bounds"]
                bounds = np.array([b_val] * current_dim)
                bounds_msg = f"Auto [{b_val[0]}, {b_val[1]}]"
            else:
                # Fallback si no hay info
                bounds = np.array([[-5.0, 5.0]] * current_dim)
                bounds_msg = "Default [-5, 5]"
        else:
            # El usuario forzó bounds específicos en config (ej: "-10, 10")
            try:
                b_min, b_max = [float(b.strip()) for b in config_bounds_str.split(',')]
                bounds = np.array([[b_min, b_max]] * current_dim)
                bounds_msg = f"Manual [{b_min}, {b_max}]"
            except:
                print("[ERROR] Bounds manuales mal formados."); sys.exit(1)

        print(f"\n>>> PROCESANDO: {obj_name.upper()} ({i_obj+1}/{len(obj_names)})")
        print(f"    Dimensión: {dim_msg} | Límites: {bounds_msg}")
        
        comparative_curves = {}
        
        for alg_name in alg_names:
            alg_fn = ALGORITMOS_REGISTRADOS.get(alg_name)
            if not alg_fn: continue
            
            alg_params = cfg.get(alg_name, {}).copy()
            
            print(f"  -> {alg_name.upper()}...", end="")
            
            run_fitnesses = []        
            all_histories = []        
            best_run_history = None   
            best_run_val = float("inf")
            
            start_time = time.time()
            
            for r in range(n_runs):
                current_seed = (base_seed + r) if base_seed is not None else None
                current_params = alg_params.copy()
                
                if animate and plot_individual:
                    current_params["log_positions"] = True
                    current_params["log_every"] = 1
                
                bx, bf, hist = alg_fn(
                    objective=objective, bounds=bounds, max_iters=base_iters,
                    pop_size=base_pop, seed=current_seed, **current_params
                )
                
                run_fitnesses.append(bf)
                if "best_f" in hist: all_histories.append(hist["best_f"])
                if bf < best_run_val:
                    best_run_val = bf
                    best_run_history = hist

            total_time = time.time() - start_time
            fit_arr = np.array(run_fitnesses)
            print(f" Fin. Mejor: {np.min(fit_arr):.2e} | Media: {np.mean(fit_arr):.2e} ({total_time:.2f}s)")

            if all_histories:
                min_len = min(len(h) for h in all_histories)
                data_mat = np.array([h[:min_len] for h in all_histories])
                comparative_curves[alg_name] = np.mean(data_mat, axis=0)

            # Gráficas Individuales (si plot_individual=True)
            if plot and plot_individual:
                if n_runs > 1:
                    plot_boxplot(run_fitnesses, title=f"{alg_name.upper()} - {obj_name}", xlabel=alg_name.upper())
                    plot_convergence_average(all_histories, title=f"Conv. {alg_name.upper()} - {obj_name}")
                elif n_runs == 1 and best_run_history:
                    plot_convergencia(best_run_history, title=f"Conv. {alg_name.upper()} - {obj_name}")

            # Animaciones Individuales (si plot_individual=True)
            if animate and plot_individual and best_run_history:
                if current_dim == 2 and "pos" in best_run_history:
                    def run_2d():
                        if show_heatmap:
                            animate_swarm_heatmap(best_run_history, objective, bounds, 
                                                title=f"{alg_name.upper()}-{obj_name}", 
                                                res=heatmap_res, levels=heatmap_levels, cmap=cmap, save_path=save_path)
                        else:
                            animate_swarm(best_run_history, bounds, title=f"{alg_name.upper()}-{obj_name}", save_path=save_path)
                    
                    def run_3d():
                        animate_swarm_3d(best_run_history, objective, bounds, 
                                         title=f"{alg_name.upper()}-{obj_name} (3D)", res=heatmap_res, cmap=cmap, save_path=save_path)

                    if animation_type == "3d": run_3d()
                    elif animation_type == "2d": run_2d()
                    elif animation_type == "both": run_3d(); run_2d()
                    else: run_2d()
                else:
                    animate_convergencia(best_run_history, title=f"Conv. {alg_name.upper()} - {obj_name}", save_path=save_path)

        # Gráfica Comparativa Final (Siempre si plot=True)
        if plot and len(comparative_curves) > 0:
            if len(comparative_curves) > 1 or not plot_individual:
                plot_comparacion_algoritmos(comparative_curves, 
                    title=f"Comparativa: {', '.join(comparative_curves.keys())} en {obj_name.upper()}", 
                    save_path=save_path)
            
    print("\n=== FIN ===")

if __name__ == "__main__":
    main()