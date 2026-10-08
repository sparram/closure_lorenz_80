import os
import torch
import numpy as np
import scipy.io
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def plot_invariant_measure(
    checkpoint_path='checkpoints/simulation_history_memory.pt',
    data_path='data/NN_training_data.mat',
    Ts=3000000,
    burn_in=1000,
    bins=80,
    cmap='turbo'
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo de simulación: {checkpoint_path}")

    print(f"Cargando historial de simulación desde {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location='cpu')

    ens_y = data['history_y'].numpy()  # Shape: [N_sim, M, 3]
    N_sim, M, _ = ens_y.shape

    if N_sim <= burn_in:
        raise ValueError(f"El número total de pasos ({N_sim}) debe ser mayor que el período de burn-in ({burn_in}).")

    # Cargar referencia real Y_true desfasada por Ts
    if os.path.exists(data_path):
        raw_mat = scipy.io.loadmat(data_path)['u']
        Y_true = raw_mat[3:6, Ts : Ts + N_sim].T  # Shape: [N_sim, 3]
    else:
        raise FileNotFoundError(f"No se encontró el archivo de referencia: {data_path}")

    # 1. Descartar el período transitorio (burn-in)
    Y_true_attractor = Y_true[burn_in:]
    history_y_attractor = ens_y[burn_in:].reshape(-1, 3)  # Aplanar todos los pasos y agentes

    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]

    # Configuración de GridSpec con 3 filas:
    # Fila 0: Real | Fila 1: CFM Approximated | Fila 2: Colorbars
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.04], hspace=0.45, wspace=0.25)

    for col, ((i, j), (lbl_i, lbl_j)) in enumerate(zip(pairs, labels)):
        # Límites comunes por columna
        x_min = min(Y_true_attractor[:, i].min(), history_y_attractor[:, i].min())
        x_max = max(Y_true_attractor[:, i].max(), history_y_attractor[:, i].max())
        y_min = min(Y_true_attractor[:, j].min(), history_y_attractor[:, j].min())
        y_max = max(Y_true_attractor[:, j].max(), history_y_attractor[:, j].max())

        xedges = np.linspace(x_min, x_max, bins + 1)
        yedges = np.linspace(y_min, y_max, bins + 1)

        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)

        # Histogramas 2D y densidad de probabilidad
        H_real, _, _ = np.histogram2d(
            Y_true_attractor[:, i], Y_true_attractor[:, j],
            bins=[xedges, yedges], density=True
        )
        H_sim, _, _ = np.histogram2d(
            history_y_attractor[:, i], history_y_attractor[:, j],
            bins=[xedges, yedges], density=True
        )

        # Suavizado gaussiano para mejor interpretabilidad visual
        H_real = ndimage.gaussian_filter(H_real, sigma=1.2)
        H_sim = ndimage.gaussian_filter(H_sim, sigma=1.2)

        vmax = max(H_real.max(), H_sim.max())
        vmin = 0.0

        # --- FILA SUPERIOR (Fila 0): Atractor Real ---
        ax_real = fig.add_subplot(gs[0, col])
        im_real = ax_real.pcolormesh(X, Y, H_real.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        ax_real.set_title(f'Real P({lbl_i}, {lbl_j})', fontsize=12, fontweight='bold')
        ax_real.set_xlabel(lbl_i)
        ax_real.set_ylabel(lbl_j)
        ax_real.grid(True, linestyle=':', alpha=0.3, color='white')

        # --- FILA INFERIOR (Fila 1): CFM Aproximado ---
        ax_sim = fig.add_subplot(gs[1, col])
        im_sim = ax_sim.pcolormesh(X, Y, H_sim.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        ax_sim.set_title(f'CFM Approximated P({lbl_i}, {lbl_j})', fontsize=12, fontweight='bold')
        ax_sim.set_xlabel(lbl_i)
        ax_sim.set_ylabel(lbl_j)
        ax_sim.grid(True, linestyle=':', alpha=0.3, color='white')

        # --- FILA DE COLORBARS (Fila 2) ---
        cax = fig.add_subplot(gs[2, col])
        cbar = fig.colorbar(im_sim, cax=cax, orientation='horizontal')
        cbar.set_label(f'Probability Density P({lbl_i}, {lbl_j})', fontsize=9)

    plt.suptitle('Closed Loop: Invariant Measure Comparison (Slow Variables Y)', fontsize=15, fontweight='bold', y=0.98)

    os.makedirs('media', exist_ok=True)
    out_path = 'media/closed_loop/invariant_measure.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"¡Gráfico guardado exitosamente en {out_path}!")
    plt.show()

if __name__ == '__main__':
    stats = torch.load('checkpoints/norm_stats_hlf.pt')
    test_start_idx = stats['split_idx']
    plot_invariant_measure(Ts=test_start_idx + 100000, cmap='turbo')