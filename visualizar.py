from __future__ import annotations
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
# --------------------------------

# ------------------------------------------
#  Animación de Enjambre 3D (Superficie)
# ------------------------------------------
def animate_swarm_3d(history: dict[str, list],
                       objective: callable,
                       bounds: np.ndarray,
                       title: str = "Animación de Enjambre 3D",
                       res: int = 100,
                       cmap: str = "viridis",
                       save_path: str | None = None):
    """
    Anima el movimiento del enjambre sobre una superficie 3D de la función objetivo.
    (Solo funciona para problemas 2D).
    """
    
    pos_hist = history.get("pos", [])
    gbest_hist = history.get("gbest_hist", [])
    if not pos_hist:
        print("Error: No hay 'pos_hist' en el historial. ¿log_positions=True?")
        return

    # --- 1. Preparar datos de la Superficie 3D ---
    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    
    # Crea la malla (mesh)
    X = np.linspace(xlo, xhi, res)
    Y = np.linspace(ylo, yhi, res)
    X, Y = np.meshgrid(X, Y)
    
    # Prepara los puntos para la evaluación
    # (res*res, 2)
    xy_pairs = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Evalúa todos los puntos de la malla
    Z = objective(xy_pairs).reshape(X.shape)
    
    z_min = Z.min()
    z_max = Z.max()

    # --- 2. Configurar la Figura 3D ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibuja la superficie de la función objetivo
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.6, antialiased=True, rcount=res, ccount=res)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(z_min, z_max)
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("Fitness")
    
    # --- 3. Configurar los Puntos de Animación ---
    
    # Inicializa el scatter plot 3D para las partículas
    # Nota: ax.scatter 3D devuelve un Path3DCollection
    scat = ax.scatter([], [], [], s=20, c="red", edgecolors="black", depthshade=True)
    
    # Inicializa el punto para el Mejor Global (Gbest)
    gdot, = ax.plot([], [], [], marker='*', linestyle="None", markersize=12, color="cyan", markeredgecolor="black")
    
    # --- 4. Definir Funciones de Animación ---
    
    def init():
        # Para 3D, 'set_data_3d' es mejor para plots, y _offsets3d para scatter
        scat._offsets3d = ([], [], [])
        gdot.set_data_3d([], [], [])
        return scat, gdot

    def update(frame):
        # Obtiene las posiciones (x, y) del historial
        pts_2d = pos_hist[frame] # (pop_size, 2)
        
        # Calcula el 'Z' (fitness) para cada partícula
        # para que se dibujen 'sobre' la superficie
        pts_z = objective(pts_2d) # (pop_size,)
        
        # Actualiza el scatter 3D
        scat._offsets3d = (pts_2d[:, 0], pts_2d[:, 1], pts_z)
        
        if gbest_hist:
            # Obtiene el Gbest (x, y)
            g_pos_2d = gbest_hist[frame] # (2,)
            
            # Calcula su 'Z' (fitness)
            g_pos_z = objective(g_pos_2d) # (,)
            
            # Actualiza el punto Gbest
            gdot.set_data_3d([g_pos_2d[0]], [g_pos_2d[1]], [g_pos_z])
            
        return scat, gdot

    # --- 5. Crear y Mostrar Animación ---
    
    # Ajusta el intervalo si hay muchos frames
    interval = max(20, 3000 // len(pos_hist))
    
    ani = FuncAnimation(
        fig,
        update,
        frames=len(pos_hist),
        init_func=init,
        blit=True,
        interval=interval,
    )

    if save_path:
        print(f"Guardando animación 3D en {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=30)
        print("Guardado.")
    else:
        plt.show()