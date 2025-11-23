import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Dict, List
from pso_config import PSOConfig

# --- Griewank ---
def griewank(x: np.ndarray) -> float:
    """
    Функція Griewank - багатомодальна тестова функція для оптимізації.
    Глобальний мінімум: f(0,...,0) = 0
    Область пошуку: x_i ∈ [-600, 600]
    """
    x = np.asarray(x)
    d = len(x)
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d+1))))
    return 1.0 + sum_term - prod_term

# --- gbest PSO ---
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

def run_multiple_experiments(
    f: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    cfg: PSOConfig,
    n_runs: int = 10
) -> Dict:
    """
    Запускає PSO n_runs разів з різними seed і збирає статистику.
    
    Повертає:
    - best_results: список найкращих значень функції для кожного прогону
    - best_positions: список найкращих позицій
    - histories: список історій збіжності
    - times: час виконання кожного прогону
    - statistics: агреговані статистики (mean, std, best, worst)
    """
    results = []
    positions = []
    histories = []
    times = []
    n_evals_list = []
    
    print(f"Запуск {n_runs} незалежних прогонів PSO...")
    print(f"Параметри: n_particles={cfg.n_particles}, w={cfg.w}, c1={cfg.c1}, c2={cfg.c2}, max_iters={cfg.max_iters}")
    print("-" * 80)
    
    for i in range(n_runs):
        # Створюємо нову конфігурацію з унікальним seed
        run_cfg = PSOConfig(
            n_particles=cfg.n_particles,
            w=cfg.w,
            c1=cfg.c1,
            c2=cfg.c2,
            max_iters=cfg.max_iters,
            seed=cfg.seed + i
        )
        
        start_time = time.time()
        result = pso_gbest(f, bounds, run_cfg)
        elapsed = time.time() - start_time
        
        results.append(result["best_f"])
        positions.append(result["best_x"])
        histories.append(result["history"])
        times.append(elapsed)
        n_evals_list.append(result["n_evals"])
        
        print(f"Прогін {i+1:2d}: f_best = {result['best_f']:.6e}, час = {elapsed:.3f}s, оцінок = {result['n_evals']}")
    
    results_array = np.array(results)
    times_array = np.array(times)
    
    statistics = {
        "best": np.min(results_array),
        "worst": np.max(results_array),
        "mean": np.mean(results_array),
        "std": np.std(results_array),
        "median": np.median(results_array),
        "mean_time": np.mean(times_array),
        "std_time": np.std(times_array),
        "mean_evals": np.mean(n_evals_list)
    }
    
    return {
        "best_results": results,
        "best_positions": positions,
        "histories": histories,
        "times": times,
        "n_evals": n_evals_list,
        "statistics": statistics
    }

def print_results(experiments: Dict, dimension: int):
    """Виводить результати експериментів у зручному форматі."""
    stats = experiments["statistics"]
    
    print("\n" + "=" * 80)
    print("ПІДСУМКОВІ РЕЗУЛЬТАТИ")
    print("=" * 80)
    print(f"Розмірність задачі: {dimension}D")
    print(f"Кількість прогонів: {len(experiments['best_results'])}")
    print()
    print("Значення цільової функції:")
    print(f"  Найкраще (best):     {stats['best']:.6e}")
    print(f"  Найгірше (worst):    {stats['worst']:.6e}")
    print(f"  Середнє (mean):      {stats['mean']:.6e}")
    print(f"  Медіана:             {stats['median']:.6e}")
    print(f"  Ст. відхилення:      {stats['std']:.6e}")
    print()
    print(f"Результат: {stats['mean']:.6e} ± {stats['std']:.6e}")
    print()
    print("Час виконання:")
    print(f"  Середній час:        {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
    print()
    print(f"Середня кількість оцінок функції: {stats['mean_evals']:.0f}")
    print()
    
    # Знаходимо індекс найкращого прогону
    best_idx = np.argmin(experiments['best_results'])
    best_position = experiments['best_positions'][best_idx]
    
    print("Найкраща знайдена позиція (оптимум):")
    print(f"  x* = {best_position}")
    print(f"  f(x*) = {experiments['best_results'][best_idx]:.6e}")
    print()
    print("Теоретичний глобальний мінімум:")
    print(f"  x* = [0, 0, ..., 0]")
    print(f"  f(x*) = 0")
    print("=" * 80)

def plot_convergence(histories: List[List[float]], title: str):
    """
    Малює криві best-so-far для всіх прогонів + середню.
    """
    max_len = max(len(h) for h in histories)
    H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
    mean_curve = H.mean(axis=0)

    plt.figure()
    for h in histories:
        plt.plot(h, alpha=0.25)
    plt.plot(mean_curve, linewidth=2, label="mean best-so-far")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far f(x)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Параметри задачі
    d = 2  # Розмірність (можна змінити на 2, 5, 10, 30 тощо)
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація PSO
    # Класичний варіант: w=0.7, c1=c2=1.5
    # Варіант Clerc: w=0.729, c1=c2=2.05 (з constriction factor χ=0.729)
    cfg = PSOConfig(
        n_particles=40,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iters=500,
        seed=42
    )
    
    print("ЛАБОРАТОРНА РОБОТА №2: PARTICLE SWARM OPTIMIZATION")
    print("Функція: Griewank")
    print(f"Область: x_i ∈ [-600, 600], розмірність: {d}D")
    print()
    
    # Запуск 10 незалежних експериментів
    experiments = run_multiple_experiments(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        n_runs=10
    )
    
    # Виведення результатів
    print_results(experiments, d)

    plot_convergence(experiments["histories"],
                 title=f"PSO convergence on Griewank ({d}D), classic")