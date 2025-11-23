from dataclasses import dataclass

# --- PSO config ---
@dataclass
class PSOConfig:
    n_particles: int = 40
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    max_iters: int = 500
    seed: int = 0