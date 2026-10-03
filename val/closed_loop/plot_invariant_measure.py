import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def plot_invariant_measure(checkpoint_path='checkpoints/simulation_history_100k.pt', burn_in=1000, bins=80, cmap='turbo'):
    print(f"Cargando historial de simulación desde {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location='cpu')
    
    history_y = data['history_y'].numpy()  # [n_steps, n_ensemble, 3]
    Y_true = data['Y_true'].numpy()        # [n_steps, 3]
    
    # 1. Descartar el período transitorio (burn-in)
    Y_true_attractor = Y_true[burn_in:]
    history_y_attractor = history_y[burn_in:].reshape(-1, 3)
    
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
        
        # Histogramas y suavizado gaussiano
        H_real, _, _ = np.histogram2d(Y_true_attractor[:, i], Y_true_attractor[:, j], 
                                      bins=[xedges, yedges], density=True)
        H_sim, _, _ = np.histogram2d(history_y_attractor[:, i], history_y_attractor[:, j], 
                                     bins=[xedges, yedges], density=True)
        
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
        
        # --- FILA INFERIOR (Fila 1): Ensamble Simulado ---
        ax_sim = fig.add_subplot(gs[1, col])
        im_sim = ax_sim.pcolormesh(X, Y, H_sim.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        ax_sim.set_title(f'CFM Approximated P({lbl_i}, {lbl_j})', fontsize=12, fontweight='bold')
        ax_sim.set_xlabel(lbl_i)
        ax_sim.set_ylabel(lbl_j)
        ax_sim.grid(True, linestyle=':', alpha=0.3, color='white')
        
        # --- FILA DE COLORBARS (Fila 2): Eje dedicado para la barra ---
        cax = fig.add_subplot(gs[2, col])
        cbar = fig.colorbar(im_sim, cax=cax, orientation='horizontal')
        cbar.set_label(f'Probability Density P({lbl_i}, {lbl_j})', fontsize=9)

    plt.suptitle('Closed Loop: Invariant Measure Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('media/invariant_measure.png', dpi=300, bbox_inches='tight')
    print("¡Gráfico guardado en media/invariant_measure.png!")
    plt.show()

if __name__ == '__main__':
    plot_invariant_measure(cmap='turbo')