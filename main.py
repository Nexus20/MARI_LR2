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

# --- PSO with Constriction Factor (Clerc & Kennedy) ---
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

def run_multiple_experiments(
    f: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    cfg: PSOConfig,
    n_runs: int = 10,
    use_constriction: bool = False,
    chi: float = 0.729
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
    
    method_name = "PSO з constriction factor (χ)" if use_constriction else "PSO зі стандартною інерцією"
    print(f"Запуск {n_runs} незалежних прогонів {method_name}...")
    if use_constriction:
        print(f"Параметри: n_particles={cfg.n_particles}, χ={chi}, c1={cfg.c1}, c2={cfg.c2}, max_iters={cfg.max_iters}")
    else:
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
        if use_constriction:
            result = pso_constriction(f, bounds, run_cfg, chi=chi)
        else:
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

def experiment_inertia(f, bounds, base_cfg: PSOConfig, w_values=(0.4, 0.7, 0.9), n_runs=10):
    summaries = {}
    histories_by_w = {}

    for w in w_values:
        cfg = PSOConfig(**{**base_cfg.__dict__, "w": w})
        exp = run_multiple_experiments(f, bounds, cfg, n_runs=n_runs)
        summaries[w] = exp
        histories_by_w[w] = exp["histories"]

    return summaries, histories_by_w

def experiment_swarm_size(f, bounds, base_cfg: PSOConfig, sizes=(20, 40, 60, 80, 100), n_runs=10):
    summaries = {}
    histories_by_n = {}

    for n in sizes:
        cfg = PSOConfig(**{**base_cfg.__dict__, "n_particles": n})
        exp = run_multiple_experiments(f, bounds, cfg, n_runs=n_runs)
        summaries[n] = exp
        histories_by_n[n] = exp["histories"]

    return summaries, histories_by_n

def experiment_coefficients(f, bounds, base_cfg: PSOConfig, c_pairs=None, n_runs=10):
    """
    Досліджує вплив когнітивних (c1) та соціальних (c2) коефіцієнтів.
    """
    if c_pairs is None:
        c_pairs = [(1.5, 1.5), (2.0, 2.0), (2.05, 2.05), (2.5, 0.5), (0.5, 2.5)]
    
    summaries = {}
    histories_by_c = {}

    for c1, c2 in c_pairs:
        cfg = PSOConfig(**{**base_cfg.__dict__, "c1": c1, "c2": c2})
        exp = run_multiple_experiments(f, bounds, cfg, n_runs=n_runs)
        summaries[(c1, c2)] = exp
        histories_by_c[(c1, c2)] = exp["histories"]

    return summaries, histories_by_c

def print_comparison_table(experiments_dict, param_name: str):
    """
    Виводить порівняльну таблицю результатів експериментів.
    
    Args:
        experiments_dict: словник {параметр: statistics}
        param_name: назва параметра для заголовку
    """
    print("\n" + "=" * 100)
    print(f"ПОРІВНЯЛЬНА ТАБЛИЦЯ: ВПЛИВ {param_name.upper()}")
    print("=" * 100)
    print(f"{'Параметр':<20} | {'Mean ± Std':<25} | {'Best':<12} | {'Worst':<12} | {'Median':<12} | {'Час (s)':<10}")
    print("-" * 100)
    
    for param, exp_data in experiments_dict.items():
        param_str = str(param)
        stats = exp_data["statistics"]
        mean_std = f"{stats['mean']:.3e} ± {stats['std']:.3e}"
        best = f"{stats['best']:.3e}"
        worst = f"{stats['worst']:.3e}"
        median = f"{stats['median']:.3e}"
        time_str = f"{stats['mean_time']:.3f}"
        
        print(f"{param_str:<20} | {mean_std:<25} | {best:<12} | {worst:<12} | {median:<12} | {time_str:<10}")
    
    print("=" * 100)

def analyze_results(experiments_dict, param_name: str):
    """
    Автоматичний аналіз результатів експериментів.
    """
    print("\n" + "=" * 80)
    print(f"АНАЛІЗ РЕЗУЛЬТАТІВ: {param_name.upper()}")
    print("=" * 80)
    
    # Знаходимо найкращі параметри за різними критеріями
    best_by_mean = min(experiments_dict.items(), key=lambda x: x[1]['statistics']['mean'])
    best_by_best = min(experiments_dict.items(), key=lambda x: x[1]['statistics']['best'])
    best_by_std = min(experiments_dict.items(), key=lambda x: x[1]['statistics']['std'])
    fastest = min(experiments_dict.items(), key=lambda x: x[1]['statistics']['mean_time'])
    
    print(f"\nНайкраще середнє значення:")
    print(f"   Параметр: {best_by_mean[0]}")
    print(f"   Mean: {best_by_mean[1]['statistics']['mean']:.6e}")
    
    print(f"\nНайкраще абсолютне значення:")
    print(f"   Параметр: {best_by_best[0]}")
    print(f"   Best: {best_by_best[1]['statistics']['best']:.6e}")
    
    print(f"\nНайстабільніший (найменше std):")
    print(f"   Параметр: {best_by_std[0]}")
    print(f"   Std: {best_by_std[1]['statistics']['std']:.6e}")
    
    print(f"\nНайшвидший:")
    print(f"   Параметр: {fastest[0]}")
    print(f"   Час: {fastest[1]['statistics']['mean_time']:.3f}s")
    
    # Висновки
    print("\nВИСНОВКИ:")
    
    # Перевірка, чи всі найкращі параметри однакові
    all_same = (best_by_mean[0] == best_by_best[0] == best_by_std[0])
    
    if all_same:
        print(f"   • Параметр {best_by_mean[0]} є оптимальним за всіма критеріями:")
        print(f"     найкраще середнє, найкращий результат та найвища стабільність.")
    else:
        if best_by_mean[0] == best_by_best[0]:
            print(f"   • Параметр {best_by_mean[0]} забезпечує найкращі результати")
            print(f"     (як середнє, так і абсолютне значення).")
        else:
            print(f"   • Параметр {best_by_mean[0]} має найкраще середнє значення.")
            print(f"   • Параметр {best_by_best[0]} досяг найкращого абсолютного результату.")
        
        if best_by_std[0] not in [best_by_mean[0], best_by_best[0]]:
            print(f"   • Параметр {best_by_std[0]} найстабільніший (мінімальне відхилення).")
    
    # Компроміс швидкість/якість
    if fastest[0] == best_by_mean[0]:
        print(f"\n   РЕКОМЕНДАЦІЯ: Використовуйте параметр {fastest[0]}")
        print(f"   (оптимальний баланс: найкраща якість + найшвидший)")
    else:
        # Порівнюємо найшвидший параметр з найкращим за якістю
        fastest_time = fastest[1]['statistics']['mean_time']
        best_time = best_by_mean[1]['statistics']['mean_time']
        fastest_quality = fastest[1]['statistics']['mean']
        best_quality = best_by_mean[1]['statistics']['mean']
        
        # Скільки разів швидше/повільніше
        if fastest_time > 0 and best_time > 0:
            time_ratio = best_time / fastest_time
        else:
            time_ratio = 1.0
            
        # Скільки разів гірше/краще за якістю (менше значення = краще)
        if fastest_quality > 0 and best_quality > 0:
            quality_ratio = fastest_quality / best_quality
        else:
            quality_ratio = 1.0
        
        print(f"\n   Компроміс швидкість/якість:")
        print(f"   • Параметр {fastest[0]} (найшвидший):")
        print(f"     - Час: {fastest_time:.3f}s (швидше в {time_ratio:.2f}x)")
        if quality_ratio > 1.01:
            print(f"     - Якість: {fastest_quality:.3e} (гірше в {quality_ratio:.2f}x)")
        elif quality_ratio < 0.99:
            print(f"     - Якість: {fastest_quality:.3e} (краще в {1/quality_ratio:.2f}x)")
        else:
            print(f"     - Якість: {fastest_quality:.3e} (приблизно однакова)")
        
        print(f"   • Параметр {best_by_mean[0]} (найкраща якість):")
        print(f"     - Час: {best_time:.3f}s (повільніше в {time_ratio:.2f}x)")
        print(f"     - Якість: {best_quality:.3e} (найкраща)")
        
        # Рекомендація залежно від пріоритету
        if time_ratio < 1.2 and quality_ratio > 1.5:
            print(f"\n   РЕКОМЕНДАЦІЯ: Використовуйте параметр {best_by_mean[0]}")
            print(f"   (незначна втрата швидкості, але суттєво краща якість)")
        elif time_ratio > 2.0 and quality_ratio < 1.2:
            print(f"\n   РЕКОМЕНДАЦІЯ: Використовуйте параметр {fastest[0]}")
            print(f"   (значний виграш у швидкості при незначній втраті якості)")
        else:
            print(f"\n   РЕКОМЕНДАЦІЯ: Вибір залежить від пріоритету:")
            print(f"   - Якщо важлива якість → {best_by_mean[0]}")
            print(f"   - Якщо важлива швидкість → {fastest[0]}")
    
    print("=" * 80)

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
    
    # Дослідження впливу інерції w
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТ 1: ВПЛИВ ІНЕРЦІЇ w НА ЗБІЖНІСТЬ")
    print("="*80)
    summaries_w, histories_w = experiment_inertia(griewank, bounds, cfg, w_values=(0.4,0.7,0.9))
    
    print_comparison_table(summaries_w, "Інерція (w)")
    analyze_results(summaries_w, "Інерція (w)")

    # графік середніх кривих для різних w
    plt.figure(figsize=(10, 6))
    for w, histories in histories_w.items():
        max_len = max(len(h) for h in histories)
        H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
        plt.plot(H.mean(axis=0), label=f"w={w}", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title("Effect of inertia weight (w) on convergence", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # Дослідження впливу розміру рою
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТ 2: ВПЛИВ РОЗМІРУ РОЮ НА ЗБІЖНІСТЬ")
    print("="*80)
    summaries_n, histories_n = experiment_swarm_size(griewank, bounds, cfg, sizes=(20,40,60,80,100))
    
    print_comparison_table(summaries_n, "Розмір рою (n_particles)")
    analyze_results(summaries_n, "Розмір рою")

    plt.figure(figsize=(10, 6))
    for n, histories in histories_n.items():
        max_len = max(len(h) for h in histories)
        H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
        plt.plot(H.mean(axis=0), label=f"n={n}", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title("Effect of swarm size on convergence", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # Дослідження впливу когнітивних і соціальних коефіцієнтів
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТ 3: ВПЛИВ КОЕФІЦІЄНТІВ c1 ТА c2 НА ЗБІЖНІСТЬ")
    print("="*80)
    c_pairs = [(1.5, 1.5), (2.0, 2.0), (2.05, 2.05), (2.5, 0.5), (0.5, 2.5)]
    summaries_c, histories_c = experiment_coefficients(griewank, bounds, cfg, c_pairs=c_pairs)
    
    print_comparison_table(summaries_c, "Коефіцієнти (c1, c2)")
    analyze_results(summaries_c, "Коефіцієнти c1, c2")

    plt.figure(figsize=(10, 6))
    for (c1, c2), histories in histories_c.items():
        max_len = max(len(h) for h in histories)
        H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
        plt.plot(H.mean(axis=0), label=f"c1={c1}, c2={c2}", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title("Effect of cognitive (c1) and social (c2) coefficients", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # ВИСОКИЙ РІВЕНЬ: Constriction Factor vs Classic PSO
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТ 4: CONSTRICTION FACTOR (CLERC) VS КЛАСИЧНИЙ PSO")
    print("="*80)
    
    # Класичний PSO з w=0.7, c1=c2=1.5
    cfg_classic = PSOConfig(
        n_particles=40,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iters=500,
        seed=42
    )
    
    # Constriction factor PSO з χ=0.729, c1=c2=2.05
    cfg_constriction = PSOConfig(
        n_particles=40,
        w=0.0,  # не використовується для constriction
        c1=2.05,
        c2=2.05,
        max_iters=500,
        seed=42
    )
    
    print("\n--- Класичний PSO (w=0.7, c1=c2=1.5) ---")
    exp_classic = run_multiple_experiments(
        f=griewank,
        bounds=bounds,
        cfg=cfg_classic,
        n_runs=10,
        use_constriction=False
    )
    
    print("\n--- Constriction Factor PSO (χ=0.729, c1=c2=2.05) ---")
    exp_constriction = run_multiple_experiments(
        f=griewank,
        bounds=bounds,
        cfg=cfg_constriction,
        n_runs=10,
        use_constriction=True,
        chi=0.729
    )
    
    # Таблиця порівняння
    comparison = {
        "Classic (w=0.7)": exp_classic,
        "Constriction (χ=0.729)": exp_constriction
    }
    
    print("\n" + "="*80)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ: CONSTRICTION FACTOR VS КЛАСИЧНИЙ PSO")
    print("="*80)
    print_comparison_table(comparison, "Метод")
    analyze_results(comparison, "Метод")
    
    # Візуалізація порівняння
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for hist in exp_classic["histories"]:
        plt.plot(hist, alpha=0.3, color='blue')
    max_len = max(len(h) for h in exp_classic["histories"])
    H_classic = np.array([h + [h[-1]]*(max_len-len(h)) for h in exp_classic["histories"]])
    plt.plot(H_classic.mean(axis=0), color='blue', linewidth=3, label='Classic (середнє)')
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Best-so-far f(x)", fontsize=11)
    plt.title("Класичний PSO (w=0.7, c1=c2=1.5)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for hist in exp_constriction["histories"]:
        plt.plot(hist, alpha=0.3, color='red')
    max_len = max(len(h) for h in exp_constriction["histories"])
    H_constr = np.array([h + [h[-1]]*(max_len-len(h)) for h in exp_constriction["histories"]])
    plt.plot(H_constr.mean(axis=0), color='red', linewidth=3, label='Constriction (середнє)')
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Best-so-far f(x)", fontsize=11)
    plt.title("Constriction Factor (χ=0.729, c1=c2=2.05)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Об'єднаний графік
    plt.figure(figsize=(10, 6))
    plt.plot(H_classic.mean(axis=0), color='blue', linewidth=2.5, label='Classic PSO (w=0.7, c1=c2=1.5)')
    plt.plot(H_constr.mean(axis=0), color='red', linewidth=2.5, label='Constriction PSO (χ=0.729, c1=c2=2.05)')
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title("Constriction Factor vs Classic PSO: Convergence Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
