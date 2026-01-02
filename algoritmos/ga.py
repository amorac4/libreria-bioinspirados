from __future__ import annotations

import numpy as np
from typing import Callable

from utils import clamp_vec_np, rand_vec_in_bounds_np, init_history

# Intentamos importar el módulo cultural (para que GA no duplique lógica)
try:
    from cultural import BeliefSpace, CulturalConfig
except Exception:
    BeliefSpace = None
    CulturalConfig = None


def ga(objective: Callable, bounds: np.ndarray,
       max_iters: int, pop_size: int, seed: int | None = None,
       pc: float = 0.9, pm: float = 0.05, elitism: int = 1,
       tournament_k: int = 3, blx_alpha: float = 0.3,
       sigma_frac: float = 0.1,

       # ===== Cultural toggle + params (desde JSON) =====
       use_cultural: bool = False,
       # Opción 1 (recomendada): pasar un dict cultural {"elite_frac":..., "p_inf":..., ...}
       cultural: dict | None = None,
       # Opción 2 (compat): pasar params sueltos (por si no usas dict)
       cultural_elite_frac: float = 0.10,
       cultural_p_inf: float = 0.25,
       cultural_beta: float = 0.15,
       cultural_ema: float = 0.30,
       cultural_min_width_frac: float = 0.05,
       # ================================================

       log_positions: bool = False, log_every: int = 1):
    """Algoritmo Genético (GA) para variables reales, versión NumPy (vectorizada).
       Si use_cultural=True, usa BeliefSpace externo (modular) para:
         - Acceptance: belief.accept(pop, fit)
         - Influence: belief.influence_batch(children, rng)
    """

    # 1) Inicialización
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    assert pop_size >= max(2, elitism + 1)

    pop = rand_vec_in_bounds_np(bounds, pop_size, rng)

    # Evaluación vectorizada
    fit = objective(pop)

    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_f = float(fit[best_idx])

    # Historial
    hist = init_history(keys=("best_f", "mean_f", "best_x", "gbest"))
    if log_positions and dim == 2:
        hist["pos"] = []

    ranges = bounds[:, 1] - bounds[:, 0]  # para mutación

    # 1.1) Construcción modular del belief space (si se activa)
    belief = None
    if use_cultural:
        if BeliefSpace is None or CulturalConfig is None:
            raise ImportError(
                "use_cultural=True pero no existe el módulo cultural. "
                "Crea cultural/config.py y cultural/belief.py (exportados en cultural/__init__.py)."
            )

        cultural = cultural or {}
        cfg = CulturalConfig(
            elite_frac=float(cultural.get("elite_frac", cultural_elite_frac)),
            p_inf=float(cultural.get("p_inf", cultural_p_inf)),
            beta=float(cultural.get("beta", cultural_beta)),
            ema=float(cultural.get("ema", cultural_ema)),
            min_width_frac=float(cultural.get("min_width_frac", cultural_min_width_frac)),
        )
        belief = BeliefSpace(bounds=bounds, cfg=cfg)
        # Acepta desde el estado inicial para poblar conocimiento desde la primera generación
        belief.accept(pop, fit)

    # 2) Bucle principal
    for it in range(max_iters):
        # --- Cultural Acceptance (hook modular) ---
        if belief is not None:
            belief.accept(pop, fit)

        # --- Elitismo ---
        elite_indices = np.argsort(fit)[:elitism]
        new_pop = pop[elite_indices].copy()

        # Crear nueva generación
        while len(new_pop) < pop_size:
            # --- Selección por Torneo (vectorizada para 2 padres) ---
            p_indices = rng.choice(pop_size, (2, tournament_k), replace=False)
            tourn_fit = fit[p_indices]
            winner_indices = p_indices[np.arange(2), np.argmin(tourn_fit, axis=1)]
            p1, p2 = pop[winner_indices]

            # --- Cruce BLX-alpha ---
            if rng.random() < pc:
                alpha = rng.uniform(-blx_alpha, 1 + blx_alpha, size=(2, dim))
                c1 = alpha[0] * p1 + (1 - alpha[0]) * p2
                c2 = alpha[1] * p1 + (1 - alpha[1]) * p2
                children = np.vstack([c1, c2])
                clamp_vec_np(children, bounds, in_place=True)
            else:
                children = np.vstack([p1, p2])

            # --- Mutación Gaussiana ---
            mutate_mask = rng.random((len(children), dim)) < pm
            if np.any(mutate_mask):
                mutations = rng.normal(0.0, sigma_frac * ranges, size=children.shape)
                children[mutate_mask] += mutations[mutate_mask]
                clamp_vec_np(children, bounds, in_place=True)

            # --- Cultural Influence (hook modular) ---
            if belief is not None:
                children = belief.influence_batch(children, rng)  # ya clampa internamente

            new_pop = np.vstack([new_pop, children])

        # Reemplazo
        pop = new_pop[:pop_size]

        # Evaluación vectorizada
        fit = objective(pop)

        # Actualizar mejor global
        current_best_idx = int(np.argmin(fit))
        if float(fit[current_best_idx]) < best_f:
            best_f = float(fit[current_best_idx])
            best_x = pop[current_best_idx].copy()

        # Historial
        hist["best_f"].append(best_f)
        hist["mean_f"].append(float(np.mean(fit)))
        hist["best_x"].append(best_x.copy())
        hist["gbest"].append(best_x.copy())
        if log_positions and dim == 2 and (it % log_every == 0):
            hist["pos"].append(pop.copy())

    return best_x, best_f, hist
