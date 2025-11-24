"""
Окремий модуль для запуску анімації PSO
Можна запустити незалежно: python run_animation.py
"""
import numpy as np
from pso_config import PSOConfig
from pso_algorithms import pso_with_trajectories
from pso_visualization import animate_pso_2d


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
    print("="*80)
    print("АНІМАЦІЯ: ТРАЄКТОРІЇ ЧАСТИНОК НА 2D ГРАФІКУ GRIEWANK")
    print("Параметри: n=40, w=0.7, c1=c2=1.5")
    print("="*80)
    
    # Параметри задачі
    d = 2  # Тільки 2D для анімації
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація PSO для анімації (використовуємо оптимальні параметри з експериментів)
    cfg_animation = PSOConfig(
        n_particles=40,  
        w=0.7,           
        c1=1.5,
        c2=1.5,
        max_iters=200,  # Менше ітерацій для швидшої анімації
        seed=42
    )
    
    # Запускаємо PSO зі збереженням траєкторій
    print("\nЗапуск PSO для анімації (збереження траєкторій)...")
    result_anim = pso_with_trajectories(
        f=griewank,
        bounds=bounds,
        cfg=cfg_animation,
        save_every=2  # Зберігати кожні 2 ітерації
    )
    
    print(f"Фінальний результат: f = {result_anim['best_f']:.6e}")
    print(f"Збережено {len(result_anim['trajectories']['particles'])} кадрів для анімації")
    
    # Створення та показ анімації
    print("\nСтворення анімації...")
    print("Підказка: для збереження у файл змініть save_path='pso_animation.gif'")
    
    # Обмежимо область для кращої візуалізації (не вся [-600, 600])
    bounds_zoom = np.array([[-100, 100], [-100, 100]])
    
    animate_pso_2d(
        trajectories=result_anim['trajectories'],
        bounds=bounds_zoom,
        f=griewank,
        save_path=None,  # Для збереження треба вказати назву файлу. Наприклад, "pso_animation.gif"
        interval=150  # Затримка між кадрами (мс)
    )
    
    print("\nАнімація завершена!")
