"""
Лабораторна робота №2: Particle Swarm Optimization
Функція: Griewank
"""
import numpy as np
import matplotlib.pyplot as plt
from pso_config import PSOConfig
from pso_algorithms import pso_with_trajectories
from pso_experiments import (
    run_multiple_experiments, 
    experiment_inertia,
    experiment_swarm_size,
    experiment_coefficients,
    print_results,
    print_comparison_table,
    analyze_results
)
from pso_visualization import plot_convergence, animate_pso_2d


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


if __name__ == "__main__":
    # Параметри задачі
    d = 2  # Розмірність (можна змінити на 2, 5, 10, 30 тощо)
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація PSO
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
    sizes = (20, 40, 60, 80, 100)
    summaries_n, histories_n = experiment_swarm_size(griewank, bounds, cfg, sizes=sizes)
    
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
    
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТИ ЗАВЕРШЕНО!")
    print("="*80)
    print("\nДля запуску анімації використайте: python run_animation.py")
