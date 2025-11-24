"""
Експерименти та аналіз результатів PSO
"""
import numpy as np
import time
from typing import Callable, Dict
from pso_config import PSOConfig
from pso_algorithms import pso_gbest, pso_constriction


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
    print(f"Середня кількість оцінок функції: {int(stats['mean_evals'])}")
    print()
    
    best_idx = np.argmin(experiments["best_results"])
    best_position = experiments["best_positions"][best_idx]
    
    print("Найкраща знайдена позиція (оптимум):")
    print(f"  x* = {best_position}")
    print(f"  f(x*) = {experiments['best_results'][best_idx]:.6e}")
    print()
    print("Теоретичний глобальний мінімум:")
    print(f"  x* = [0, 0, ..., 0]")
    print(f"  f(x*) = 0")
    print("=" * 80)


def print_comparison_table(experiments_dict, param_name: str):
    """
    Виводить порівняльну таблицю результатів експериментів.
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
