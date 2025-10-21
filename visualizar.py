# visualizar.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("iteración")
    ax.set_ylabel("best_f")

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
def animate_swarm(history: dict[str, list],
                  bounds: np.ndarray, # Acepta np.ndarray
                  interval_ms: int = 60,
                  title: str = "Swarm (2D)",
                  show_gbest_path: bool = True,
                  save_path: str | None = None):
    pos_hist = history.get("pos")
    gbest_hist = history.get("gbest")
    if not pos_hist:
        raise ValueError("No hay posiciones en history['pos']. Ejecuta con log_positions=True y dim=2.")

    # bounds ahora es un array (dim, 2)
    xlo, xhi = bounds[0, 0], bounds[0, 1]
    ylo, yhi = bounds[1, 0], bounds[1, 1]
    
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=25)
    gdot, = ax.plot([], [], marker="x", linestyle="none", markersize=8)  # gbest actual
    gpath = None
    if show_gbest_path:
        gpath, = ax.plot([], [], lw=1.2, alpha=0.8)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gdot.set_data([], [])
        if gpath is not None:
            gpath.set_data([], [])
        return (scat, gdot, gpath) if gpath is not None else (scat, gdot)

    def update(frame):
        # pos_hist ya contiene arrays de NumPy (pop_size, dim)
        pts = pos_hist[frame] 
        scat.set_offsets(pts[:, :2]) # Asegura que solo tome x, y

        if gbest_hist:
            g_vec = gbest_hist[frame]
            gdot.set_data([g_vec[0]], [g_vec[1]])
            if gpath is not None:
                # gbest_hist es una lista de np.ndarray
                g_path_data = np.array(gbest_hist[: frame + 1])
                gpath.set_data(g_path_data[:, 0], g_path_data[:, 1])

        return (scat, gdot, gpath) if gpath is not None else (scat, gdot)

    ani = FuncAnimation(fig, update, frames=len(pos_hist), init_func=init, interval=interval_ms, blit=True)
    plt.tight_layout()
    if save_path:
        ani.save(save_path)
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
                          objective,
                          bounds: np.ndarray, # Acepta np.ndarray
                          interval_ms: int = 60,
                          title: str = "Swarm (2D) + Heatmap",
                          show_gbest_path: bool = True,
                          res: int = 200, levels: int = 30, cmap: str = "viridis",
                          save_path: str | None = None):
    pos_hist = history.get("pos")
    gbest_hist = history.get("gbest")
    if not pos_hist:
        raise ValueError("No hay posiciones en history['pos']. Ejecuta con log_positions=True y dim=2.")

    xlo, xhi = bounds[0, 0], bounds[0, 1]
    ylo, yhi = bounds[1, 0], bounds[1, 1]
    X, Y, Z = _grid_objective(objective, bounds, res=res)

    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.95)
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    fig.colorbar(cs, ax=ax)

    scat = ax.scatter([], [], s=25, c="white", edgecolors="black", linewidths=0.5)
    gdot, = ax.plot([], [], marker="x", linestyle="none", markersize=8, color="red")
    gpath = None
    if show_gbest_path:
        gpath, = ax.plot([], [], lw=1.2, alpha=0.9, color="red")

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gdot.set_data([], [])
        if gpath is not None:
            gpath.set_data([], [])
        return (scat, gdot, gpath) if gpath is not None else (scat, gdot)

    def update(frame):
        # pos_hist ya contiene arrays de NumPy (pop_size, dim)
        pts = pos_hist[frame]
        scat.set_offsets(pts[:, :2])

        if gbest_hist:
            g_vec = gbest_hist[frame]
            gdot.set_data([g_vec[0]], [g_vec[1]])
            if gpath is not None:
                g_path_data = np.array(gbest_hist[: frame + 1])
                gpath.set_data(g_path_data[:, 0], g_path_data[:, 1])

        return (scat, gdot, gpath) if gpath is not None else (scat, gdot)

    ani = FuncAnimation(fig, update, frames=len(pos_hist), init_func=init, interval=interval_ms, blit=True)
    plt.tight_layout()
    if save_path:
        ani.save(save_path)
    else:
        plt.show()