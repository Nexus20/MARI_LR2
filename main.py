import numpy as np
from typing import Callable, Dict
from pso_config import PSOConfig

# --- Griewank ---
def griewank(x: np.ndarray) -> float:
    x = np.asarray(x)
    d = len(x)
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d+1))))
    return 1.0 + sum_term - prod_term

# --- gbest PSO ---
def pso_gbest(f: Callable[[np.ndarray], float], bounds: np.ndarray, cfg: PSOConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    X = lo + rng.random((cfg.n_particles, D)) * span
    V = rng.uniform(-span, span, size=(cfg.n_particles, D)) * 0.1

    pbest_X = X.copy()
    pbest_f = np.array([f(x) for x in X])

    g_idx = np.argmin(pbest_f)
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])

    for _ in range(cfg.max_iters):
        r1 = rng.random((cfg.n_particles, D))
        r2 = rng.random((cfg.n_particles, D))

        V = (cfg.w * V
             + cfg.c1 * r1 * (pbest_X - X)
             + cfg.c2 * r2 * (gbest_X - X))

        X = X + V
        X = np.clip(X, lo, hi)  # просте утримання області

        f_vals = np.array([f(x) for x in X])
        improve = f_vals < pbest_f
        pbest_X[improve] = X[improve]
        pbest_f[improve] = f_vals[improve]

        g_idx = np.argmin(pbest_f)
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest_X = pbest_X[g_idx].copy()

    return {"best_x": gbest_X, "best_f": gbest_f}

if __name__ == "__main__":
    d = 2  # можна змінити на іншу розмірність
    bounds = np.array([[-600, 600]] * d)

    cfg = PSOConfig(n_particles=40, w=0.7, c1=1.5, c2=1.5, max_iters=500, seed=2)

    res = pso_gbest(griewank, bounds, cfg)
    print("best_x:", res["best_x"])
    print("best_f:", res["best_f"])