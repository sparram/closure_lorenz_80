import os
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def plot_survival_function(
    checkpoint_path='checkpoints/simulation_history_memory.pt',
    data_path='data/NN_training_data.mat',
    Ts=3000000
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo de simulación: {checkpoint_path}")

    print(f"Cargando simulación desde: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    ens_y = data['history_y'].numpy()  # Shape: [N_sim, M, 3]
    N_sim, M, _ = ens_y.shape

    # Aplanamos todas las simulaciones y pasos del ensemble para obtener la distribución estadística global
    ens_y_flat = ens_y.reshape(-1, 3)

    # Cargar datos reales de referencia partiendo del mismo instante Ts
    if os.path.exists(data_path):
        raw_mat = scipy.io.loadmat(data_path)['u']
        # Tomamos una longitud de referencia equivalente o completa
        Y_true = raw_mat[3:6, Ts : Ts + N_sim].T  # Shape: [N_sim, 3]
    else:
        raise FileNotFoundError(f"No se encontró el archivo de datos de referencia: {data_path}")

    # --- FUNCIÓN PARA CALCULAR LA CCDF (Probabilidad de Excedencia: P(Y > y)) ---
    def get_ccdf(arr):
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        # CCDF empírica: fracción de puntos estrictamente mayores o iguales
        ccdf = 1.0 - (np.arange(1, n + 1) / n)
        return sorted_arr, ccdf

    # --- CONFIGURACIÓN DE ESTILO ---
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })

    # --- GRAFICACIÓN DE LAS COLAS (3 Subplots para Y1, Y2, Y3) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['$Y_1$', '$Y_2$', '$Y_3$']

    for idx in range(3):
        # Calcular CCDF para datos reales y del modelo CFM
        x_real, y_real = get_ccdf(Y_true[:, idx])
        x_cfm, y_cfm = get_ccdf(ens_y_flat[:, idx])

        # Gráfica semilogarítmica para resaltar las colas (eventos raros)
        axes[idx].semilogy(x_real, y_real, 'k-', label='Real Data', linewidth=1.5, alpha=0.9)
        axes[idx].semilogy(x_cfm, y_cfm, 'r--', label='CFM Ensemble', linewidth=1.5, alpha=0.8)

        axes[idx].set_title(f'Tail Distribution: {labels[idx]}')
        axes[idx].set_xlabel(f'{labels[idx]} value ($y$)')
        axes[idx].set_ylabel('$P(Y > y)$ (Log Scale)')
        axes[idx].legend(loc='upper right', frameon=True)
        axes[idx].grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()

    os.makedirs('media/closed_loop', exist_ok=True)
    out_path = "media/closed_loop/survival_function_tails.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig("media/closed_loop/survival_function_tails.pdf", bbox_inches='tight')
    print(f"¡Gráfica de función de supervivencia guardada con éxito en {out_path}!")
    plt.close()

if __name__ == '__main__':
    stats = torch.load('checkpoints/norm_stats_hlf.pt', weights_only=True)
    test_start_idx = stats['split_idx']
    plot_survival_function(Ts=test_start_idx + 230000)