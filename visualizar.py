from __future__ import annotations
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
#  Curva de convergencia (plot)
# -----------------------------
def plot_convergencia(history: dict[str, list], title: str = "Convergencia"):
    if "best_f" not in history:
        return
    plt.figure()
    plt.plot(history["best_f"], label="best_f")
    if "mean_f" in history:
        plt.plot(history["mean_f"], label="mean_f", linestyle=":", alpha=0.8)
    plt.title(title)
    plt.xlabel("iteración")
    plt.ylabel("fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------
#  Curva de convergencia (animada incremental)
# ------------------------------------------
def animate_convergencia(history: dict[str, list], interval_ms: int = 60,
                         title: str = "Convergencia (animada)",
                         save_path: str | None = None):
    y = history.get("best_f", [])
    if not y:
        return
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlim(0, len(y))
    ymin, ymax = float(min(y)), float(max(y))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("iteración")
    ax.set_ylabel("Mejor fitness")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        xs = list(range(frame + 1))
        ys = y[: frame + 1]
        line.set_data(xs, ys)
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(y), init_func=init, interval=interval_ms, blit=True)
    plt.tight_layout()
    if save_path:
        ani.save(save_path)
    else:
        plt.show()

# --------------------------------
#  Animación del enjambre (2D puro)
# --------------------------------

def animate_swarm(history: dict[str, list], bounds: np.ndarray,
                  title: str = "Animación de Enjambre",
                  save_path: str | None = None):
    
    pos_hist = history.get("pos", [])
    if not pos_hist:
        return

    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]

    fig, ax = plt.subplots()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    # Título inicial (se actualizará)
    ax.set_title(title)

    scat = ax.scatter([], [], c="black", s=20, alpha=0.6)
    gdot, = ax.plot([], [], marker="x", linestyle="none", markersize=10, color="red")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gdot.set_data([], [])
        return scat, gdot

    def update(frame):
        # Posiciones actuales
        current_pos = pos_hist[frame]
        scat.set_offsets(current_pos)
        
        # Mejor global actual
        gbest_curr = history["gbest_hist"][frame]
        gdot.set_data([gbest_curr[0]], [gbest_curr[1]])
        
        # --- CAMBIO AQUÍ: Actualizar el título con la iteración ---
        ax.set_title(f"{title}\nIteración: {frame}/{len(pos_hist)-1}")
        
        return scat, gdot

    ani = FuncAnimation(fig, update, frames=len(pos_hist),
                        init_func=init, blit=False, interval=50) # blit=False para que el título se actualice bien

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=15)
    else:
        plt.show()



# ---------------------------------------------------
#  Animación del enjambre (2D) con heatmap del objetivo
# ---------------------------------------------------
def _grid_objective(objective, bounds: np.ndarray, res: int = 200):
    """ Genera la malla de evaluación para el heatmap, de forma vectorizada. """
    (xlo, xhi), (ylo, yhi) = bounds[:2] # Solo para 2D
    xs = np.linspace(xlo, xhi, res)
    ys = np.linspace(ylo, yhi, res)
    X, Y = np.meshgrid(xs, ys)
    
    # Crea una matriz de puntos (res, res, 2) para evaluar el objetivo de una sola vez
    grid_points = np.stack([X, Y], axis=-1)
    Z = objective(grid_points)
    return X, Y, Z


def animate_swarm_heatmap(history: dict[str, list],
                          objective: Callable,
                          bounds: np.ndarray,
                          title: str = "Mapa de Calor",
                          res: int = 100,
                          levels: int = 20,
                          cmap: str = "viridis",
                          save_path: str | None = None,
                          show_gbest_path: bool = False):
    
    pos_hist = history.get("pos", [])
    gbest_hist = history.get("gbest_hist", [])
    if not pos_hist:
        print("No hay historial de posiciones ('pos') para animar.")
        return

    # Configurar malla
    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    X = np.linspace(xlo, xhi, res)
    Y = np.linspace(ylo, yhi, res)
    X, Y = np.meshgrid(X, Y)
    
    # Evaluar Z
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = objective(XY).reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.95)
    fig.colorbar(cs, ax=ax)

    scat = ax.scatter([], [], s=25, c="white", edgecolors="black", linewidths=0.5)
    gdot, = ax.plot([], [], marker="x", linestyle="none", markersize=8, color="red")
    
    gpath = None
    if show_gbest_path:
        gpath, = ax.plot([], [], lw=1.2, alpha=0.9, color="red")

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gdot.set_data([], [])
        if gpath is not None:
            gpath.set_data([], [])
        # Retorno condicional para evitar errores
        items = [scat, gdot]
        if gpath: items.append(gpath)
        return tuple(items)

    def update(frame):
        pts = pos_hist[frame]
        scat.set_offsets(pts[:, :2])

        if gbest_hist:
            gb = gbest_hist[frame]
            gdot.set_data([gb[0]], [gb[1]])
            
            if gpath is not None:
                # Rastro completo hasta el frame actual
                hist_arr = np.array(gbest_hist[:frame+1])
                gpath.set_data(hist_arr[:, 0], hist_arr[:, 1])

        # --- CAMBIO AQUÍ: Actualizar título ---
        ax.set_title(f"{title}\nIteración: {frame}/{len(pos_hist)-1}")

        items = [scat, gdot]
        if gpath: items.append(gpath)
        return tuple(items)

    # blit=False es importante para que los cambios de título (fuera del canvas del plot) se rendericen
    ani = FuncAnimation(fig, update, frames=len(pos_hist),
                        init_func=init, blit=False, interval=60)

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=15)
    else:
        plt.show()


# --------------------------------

# ------------------------------------------
#  Animación de Enjambre 3D (Superficie)
# ------------------------------------------

def animate_swarm_3d(history: dict, objective: Callable, bounds: np.ndarray,
                     title: str = "Swarm 3D", res: int = 50, cmap: str = 'viridis',
                     save_path: str | None = None):
    """
    Animación 3D optimizada: Dibuja la superficie UNA vez y solo mueve los puntos.
    """
    if "pos" not in history or not history["pos"]:
        print("[Visualizar] No hay historial de posiciones ('pos') para animar.")
        return

    positions_history = history["pos"] # Lista de arrays (pop_size, dim)
    
    # 1. Preparar la Superficie (SOLO UNA VEZ)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generar malla
    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]
    
    # Usamos 'res' controlado (ej. 50) para no saturar
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Evaluar Z
    # Aplanamos para evaluar vectorizado y luego volvemos a la forma de malla
    points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    Z = objective(points).reshape(X_grid.shape)
    
    # Dibujar superficie estática (alpha bajo para ver puntos detrás)
    ax.plot_surface(X_grid, Y_grid, Z, cmap=cmap, alpha=0.6, edgecolor='none')
    
    # Configurar límites y etiquetas
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    # 2. Inicializar los puntos (Scatter)
    # Inicialmente vacíos, se llenan en el update
    scat = ax.scatter([], [], [], c='red', s=40, depthshade=False, edgecolors='black')
    
    # Texto de iteración
    txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # 3. Función de Actualización (SOLO mueve puntos)
    def update(frame):
        # Obtener posiciones de la iteración actual
        pop = positions_history[frame] # (N, dim)
        
        # Calcular altura Z para cada partícula (para que estén sobre la superficie)
        # Opcional: Si quieres que los puntos floten en su valor real de fitness:
        z_vals = objective(pop) 
        
        # Actualizar datos del scatter
        scat._offsets3d = (pop[:, 0], pop[:, 1], z_vals)
        
        txt.set_text(f"Iteración: {frame}")
        return scat, txt

    # Crear animación
    anim = FuncAnimation(fig, update, frames=len(positions_history),
                         interval=100, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animación 3D guardada en: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
# ------------------------------------------
#  Gráficas Estadísticas (Boxplot y Promedio)
# ------------------------------------------

def plot_boxplot(data: list[float] | np.ndarray, 
                 title: str = "Distribución de Fitness Final",
                 xlabel: str = "Algoritmo",
                 save_path: str | None = None):
    """
    Genera un diagrama de caja (Boxplot) de los valores finales de fitness.
    Ayuda a visualizar la estabilidad del algoritmo.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Crear el boxplot
    ax.boxplot(data, patch_artist=True, 
               boxprops=dict(facecolor="lightblue", color="blue"),
               medianprops=dict(color="red", linewidth=2))
    
    ax.set_title(title)
    ax.set_ylabel("Fitness (Mejor Valor)")
    ax.set_xticklabels([xlabel])
    ax.grid(True, linestyle="--", alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfica guardada en: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_convergence_average(all_runs_history: list[list[float]], 
                             title: str = "Convergencia Promedio",
                             save_path: str | None = None):
    """
    Grafica la curva promedio de convergencia con sombra de desviación estándar.
    Recibe una lista de listas, donde cada sub-lista es el historial 'best_f' de una corrida.
    """
    if not all_runs_history:
        return

    # Convertir a matriz numpy: (n_runs, n_iters)
    # Nota: Asumimos que todas las corridas tienen el mismo número de iteraciones.
    min_len = min(len(h) for h in all_runs_history)
    data = np.array([h[:min_len] for h in all_runs_history])
    
    iters = np.arange(min_len)
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Línea promedio
    ax.plot(iters, mean_curve, label="Promedio", color="blue", linewidth=2)
    
    # Sombra de desviación estándar (Promedio ± Std)
    ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, 
                    color="blue", alpha=0.2, label="Desviación Estándar")
    
    ax.set_title(title)
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Fitness (Log scale)" if np.min(mean_curve) > 0 else "Fitness")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    # Opcional: Escala logarítmica si los valores son muy dispares
    # ax.set_yscale("log")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfica guardada en: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_comparacion_algoritmos(data_dict: dict[str, list[float] | np.ndarray],
                                title: str = "Comparación de Algoritmos",
                                save_path: str | None = None):
    """
    Grafica múltiples curvas de convergencia en la misma figura.
    
    Args:
        data_dict: Diccionario { "NombreAlgoritmo": [lista_de_fitness_promedio] }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    styles = ['-', '--', '-.', ':']
    markers = [None, 'o', 's', '^', 'v', 'x']
    
    for i, (alg_name, curve) in enumerate(data_dict.items()):
        # Estilo rotativo para diferenciar líneas
        st = styles[i % len(styles)]
        # mk = markers[i % len(markers)] # Descomentar si quieres marcadores (a veces ensucia la gráfica)
        
        # Graficar
        iters = range(len(curve))
        ax.plot(iters, curve, label=alg_name, linestyle=st, linewidth=2)

    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Fitness (Promedio)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    # Escala logarítmica suele ser mejor para ver diferencias pequeñas cerca de cero
    # ax.set_yscale('log') 

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfica comparativa guardada en: {save_path}")
    else:
        plt.show()
    plt.close()