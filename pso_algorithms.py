"""
PSO алгоритми: різні варіанти Particle Swarm Optimization
"""
import numpy as np
from typing import Callable, Dict
from pso_config import PSOConfig


def pso_gbest(f: Callable[[np.ndarray], float], bounds: np.ndarray, cfg: PSOConfig) -> Dict:
    """
    Particle Swarm Optimization з глобальною топологією (gbest).
    
    Формула оновлення швидкості:
    v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i(t)) + c2·r2·(gbest - x_i(t))
    
    Параметри:
    - w (інерція): керує балансом між exploration/exploitation
    - c1 (когнітивна): вплив особистого найкращого результату
    - c2 (соціальна): вплив глобального найкращого результату
    """
    rng = np.random.default_rng(cfg.seed)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # Ініціалізація позицій та швидкостей
    X = lo + rng.random((cfg.n_particles, D)) * span
    V = rng.uniform(-span, span, size=(cfg.n_particles, D)) * 0.1

    # Ініціалізація особистих найкращих позицій
    pbest_X = X.copy()
    pbest_f = np.array([f(x) for x in X])

    # Ініціалізація глобального найкращого
    g_idx = np.argmin(pbest_f)
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    
    # Історія збіжності
    history = [gbest_f]
    n_evals = cfg.n_particles  # Лічильник оцінок функції

    for iteration in range(cfg.max_iters):
        # Генерація випадкових коефіцієнтів
        r1 = rng.random((cfg.n_particles, D))
        r2 = rng.random((cfg.n_particles, D))

        # Оновлення швидкості (основна формула PSO)
        V = (cfg.w * V
             + cfg.c1 * r1 * (pbest_X - X)
             + cfg.c2 * r2 * (gbest_X - X))

        # Оновлення позицій
        X = X + V
        X = np.clip(X, lo, hi)  # Утримання частинок в межах області

        # Оцінка функції для всіх частинок
        f_vals = np.array([f(x) for x in X])
        n_evals += cfg.n_particles
        
        # Оновлення особистих найкращих позицій
        improve = f_vals < pbest_f
        pbest_X[improve] = X[improve]
        pbest_f[improve] = f_vals[improve]

        # Оновлення глобального найкращого
        g_idx = np.argmin(pbest_f)
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest_X = pbest_X[g_idx].copy()
        
        history.append(gbest_f)

    return {
        "best_x": gbest_X, 
        "best_f": gbest_f,
        "history": history,
        "n_evals": n_evals
    }


def pso_with_trajectories(f: Callable[[np.ndarray], float], bounds: np.ndarray, cfg: PSOConfig, 
                          save_every: int = 5) -> Dict:
    """
    PSO з збереженням траєкторій частинок для анімації.
    
    Параметри:
    - save_every: зберігати позиції кожні N ітерацій (для зменшення пам'яті)
    
    Повертає позиції частинок, pbest, gbest для кожного збереженого кроку.
    """
    rng = np.random.default_rng(cfg.seed)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # Ініціалізація
    X = lo + rng.random((cfg.n_particles, D)) * span
    V = rng.uniform(-span, span, size=(cfg.n_particles, D)) * 0.1
    pbest_X = X.copy()
    pbest_f = np.array([f(x) for x in X])
    
    g_idx = np.argmin(pbest_f)
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    
    # Збереження траєкторій
    trajectories = {
        'particles': [X.copy()],  # Позиції всіх частинок
        'pbest': [pbest_X.copy()],  # Особисті найкращі
        'gbest': [gbest_X.copy()],  # Глобальний найкращий
        'gbest_f': [gbest_f],  # Значення gbest
        'iterations': [0]  # Номери ітерацій
    }
    
    history = [gbest_f]
    n_evals = cfg.n_particles

    for iteration in range(1, cfg.max_iters + 1):
        r1 = rng.random((cfg.n_particles, D))
        r2 = rng.random((cfg.n_particles, D))

        V = (cfg.w * V + cfg.c1 * r1 * (pbest_X - X) + cfg.c2 * r2 * (gbest_X - X))
        X = X + V
        X = np.clip(X, lo, hi)

        f_vals = np.array([f(x) for x in X])
        n_evals += cfg.n_particles
        
        improve = f_vals < pbest_f
        pbest_X[improve] = X[improve]
        pbest_f[improve] = f_vals[improve]

        g_idx = np.argmin(pbest_f)
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest_X = pbest_X[g_idx].copy()
        
        history.append(gbest_f)
        
        # Зберігаємо кожні save_every ітерацій
        if iteration % save_every == 0 or iteration == cfg.max_iters:
            trajectories['particles'].append(X.copy())
            trajectories['pbest'].append(pbest_X.copy())
            trajectories['gbest'].append(gbest_X.copy())
            trajectories['gbest_f'].append(gbest_f)
            trajectories['iterations'].append(iteration)

    return {
        "best_x": gbest_X,
        "best_f": gbest_f,
        "history": history,
        "n_evals": n_evals,
        "trajectories": trajectories
    }


def pso_constriction(f: Callable[[np.ndarray], float], bounds: np.ndarray, cfg: PSOConfig, chi: float = 0.729) -> Dict:
    """
    Particle Swarm Optimization з constriction factor (χ) за Clerc & Kennedy.
    
    Формула оновлення швидкості:
    v_i(t+1) = χ · [v_i(t) + c1·r1·(pbest_i - x_i(t)) + c2·r2·(gbest - x_i(t))]
    
    Constriction factor χ забезпечує збіжність без потреби в явному обмеженні швидкості.
    Типові значення: χ = 0.729, c1 = c2 = 2.05
    
    Параметри:
    - chi (χ): constriction factor, зазвичай 0.729
    - c1, c2: когнітивна та соціальна константи, зазвичай 2.05
    """
    rng = np.random.default_rng(cfg.seed)
    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # Ініціалізація позицій та швидкостей
    X = lo + rng.random((cfg.n_particles, D)) * span
    V = rng.uniform(-span, span, size=(cfg.n_particles, D)) * 0.1

    # Ініціалізація особистих найкращих позицій
    pbest_X = X.copy()
    pbest_f = np.array([f(x) for x in X])

    # Ініціалізація глобального найкращого
    g_idx = np.argmin(pbest_f)
    gbest_X = pbest_X[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    
    # Історія збіжності
    history = [gbest_f]
    n_evals = cfg.n_particles

    for iteration in range(cfg.max_iters):
        # Генерація випадкових коефіцієнтів
        r1 = rng.random((cfg.n_particles, D))
        r2 = rng.random((cfg.n_particles, D))

        # Оновлення швидкості з constriction factor
        V = chi * (V + cfg.c1 * r1 * (pbest_X - X) + cfg.c2 * r2 * (gbest_X - X))

        # Оновлення позицій
        X = X + V
        X = np.clip(X, lo, hi)

        # Оцінка функції для всіх частинок
        f_vals = np.array([f(x) for x in X])
        n_evals += cfg.n_particles
        
        # Оновлення особистих найкращих позицій
        improve = f_vals < pbest_f
        pbest_X[improve] = X[improve]
        pbest_f[improve] = f_vals[improve]

        # Оновлення глобального найкращого
        g_idx = np.argmin(pbest_f)
        if pbest_f[g_idx] < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest_X = pbest_X[g_idx].copy()
        
        history.append(gbest_f)

    return {
        "best_x": gbest_X, 
        "best_f": gbest_f,
        "history": history,
        "n_evals": n_evals,
        "chi": chi
    }
