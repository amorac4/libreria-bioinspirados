import numpy as np
from .config import CulturalConfig

class BeliefSpace:
    """
    Belief Space genérico para optimización continua.
    - Situational knowledge: mejor ejemplar observado (best_x)
    - Normative knowledge: intervalos por gen (norm_lb/norm_ub)
    """

    def __init__(self, bounds: np.ndarray, cfg: CulturalConfig):
        self.bounds = np.asarray(bounds, dtype=float)
        self.cfg = cfg
        self.dim = self.bounds.shape[0]
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]

        self.norm_lb = self.bounds[:, 0].copy()
        self.norm_ub = self.bounds[:, 1].copy()

        self.best_x = None
        self.best_f = np.inf

    def accept(self, pop: np.ndarray, fit: np.ndarray):
        """Acceptance: actualiza belief space usando la élite."""
        n = pop.shape[0]
        e = max(2, int(np.ceil(self.cfg.elite_frac * n)))
        elite_idx = np.argsort(fit)[:e]
        elite = pop[elite_idx]
        elite_fit = fit[elite_idx]

        # Situational: mejor de la élite
        j = int(np.argmin(elite_fit))
        if float(elite_fit[j]) < self.best_f:
            self.best_f = float(elite_fit[j])
            self.best_x = elite[j].copy()

        # Normative: min/max de la élite + EMA
        new_lb = np.min(elite, axis=0)
        new_ub = np.max(elite, axis=0)

        a = self.cfg.ema
        self.norm_lb = (1 - a) * self.norm_lb + a * new_lb
        self.norm_ub = (1 - a) * self.norm_ub + a * new_ub

        # Respeta bounds globales
        self.norm_lb = np.maximum(self.norm_lb, self.bounds[:, 0])
        self.norm_ub = np.minimum(self.norm_ub, self.bounds[:, 1])

        # Evita colapso: ancho mínimo
        min_w = self.cfg.min_width_frac * self.ranges
        w = self.norm_ub - self.norm_lb
        too_narrow = w < min_w
        if np.any(too_narrow):
            mid = 0.5 * (self.norm_lb + self.norm_ub)
            half = 0.5 * min_w
            self.norm_lb = np.where(too_narrow, mid - half, self.norm_lb)
            self.norm_ub = np.where(too_narrow, mid + half, self.norm_ub)
            self.norm_lb = np.maximum(self.norm_lb, self.bounds[:, 0])
            self.norm_ub = np.minimum(self.norm_ub, self.bounds[:, 1])

    def influence_vec(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Influence para un vector (candidato/partícula)."""
        y = x.copy()

        # 1) Atracción a mejor ejemplar
        if self.best_x is not None and self.cfg.beta != 0:
            y = y + self.cfg.beta * (self.best_x - y)

        # 2) Re-muestreo por gen en rango normativo
        if self.cfg.p_inf > 0:
            mask = rng.random(self.dim) < self.cfg.p_inf
            if np.any(mask):
                y[mask] = rng.uniform(self.norm_lb[mask], self.norm_ub[mask])

        return np.clip(y, self.bounds[:, 0], self.bounds[:, 1])

    def influence_batch(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Influence para lote (población / partículas)."""
        Y = X.copy()

        if self.best_x is not None and self.cfg.beta != 0:
            Y = Y + self.cfg.beta * (self.best_x - Y)

        if self.cfg.p_inf > 0:
            mask = rng.random(Y.shape) < self.cfg.p_inf
            if np.any(mask):
                samples = rng.uniform(self.norm_lb, self.norm_ub, size=Y.shape)
                Y[mask] = samples[mask]

        return np.clip(Y, self.bounds[:, 0], self.bounds[:, 1])
