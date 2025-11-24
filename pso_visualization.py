"""
Візуалізація результатів PSO: графіки збіжності та анімація траєкторій
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Callable, Dict, List


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


def animate_pso_2d(trajectories: Dict, bounds: np.ndarray, f: Callable, 
                   save_path: str = None, interval: int = 100):
    """
    Створює анімацію руху частинок PSO на 2D графіку функції.
    
    Параметри:
    - trajectories: словник з траєкторіями від pso_with_trajectories()
    - bounds: межі області пошуку
    - f: функція для відображення контурів
    - save_path: шлях для збереження gif (якщо None - тільки показати)
    - interval: затримка між кадрами в мс
    """
    # Створення сітки для контурного графіку
    n_points = 200
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Обчислення значень функції
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))
    
    # Налаштування графіків
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Лівий графік: траєкторії частинок
    contour = ax1.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.7)
    ax1.contour(X1, X2, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='f(x)')
    
    # Маркери для частинок та gbest
    particles_scatter = ax1.scatter([], [], c='red', s=30, alpha=0.6, label='Particles')
    gbest_scatter = ax1.scatter([], [], c='yellow', s=200, marker='*', 
                                edgecolors='black', linewidths=2, label='Global Best', zorder=10)
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('PSO Particle Trajectories on Griewank Function', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Правий графік: збіжність
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best-so-far f(x)', fontsize=12)
    ax2.set_title('Convergence History', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    convergence_line, = ax2.plot([], [], 'b-', linewidth=2)
    convergence_point = ax2.scatter([], [], c='red', s=100, zorder=5)
    
    # Текст для відображення інформації
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        gbest_scatter.set_offsets(np.empty((0, 2)))
        convergence_line.set_data([], [])
        convergence_point.set_offsets(np.empty((0, 2)))
        info_text.set_text('')
        return particles_scatter, gbest_scatter, convergence_line, convergence_point, info_text
    
    def update(frame):
        # Позиції частинок
        particles = trajectories['particles'][frame]
        particles_scatter.set_offsets(particles)
        
        # Глобальний найкращий
        gbest = trajectories['gbest'][frame]
        gbest_scatter.set_offsets(gbest.reshape(1, -1))
        
        # Оновлення графіку збіжності
        iter_num = trajectories['iterations'][frame]
        gbest_values = trajectories['gbest_f'][:frame+1]
        iters = trajectories['iterations'][:frame+1]
        convergence_line.set_data(iters, gbest_values)
        convergence_point.set_offsets([[iter_num, gbest_values[-1]]])
        
        # Автомасштабування для графіку збіжності
        if frame == 0:
            ax2.set_xlim(0, trajectories['iterations'][-1])
            ax2.set_ylim(min(trajectories['gbest_f']) * 0.5, 
                        max(trajectories['gbest_f']) * 2)
        
        # Інформаційний текст
        info_text.set_text(f"Iteration: {iter_num}\n"
                          f"Global Best: {trajectories['gbest_f'][frame]:.6e}\n"
                          f"Particles: {len(particles)}")
        
        return particles_scatter, gbest_scatter, convergence_line, convergence_point, info_text
    
    n_frames = len(trajectories['particles'])
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        print(f"Збереження анімації у {save_path}...")
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        print("Анімацію збережено!")
    
    plt.tight_layout()
    plt.show()
    
    return anim
