from dataclasses import dataclass

@dataclass
class CulturalConfig:
    elite_frac: float = 0.1  # Fraction of elite individuals
    p_inf: float = 0.25     # Probability of influence
    beta: float = 0.5       # Learning rate
    ema: float = 0.9        # Exponential moving average factor
    min_width_frac: float = .05   # Minimum width for belief space
    