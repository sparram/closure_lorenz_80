import os
import torch
import scipy.io
import matplotlib.pyplot as plt

def plot_attractor(
    checkpoint_path='checkpoints/simulation_history_memory.pt',
    data_path='data/NN_training_data.mat',
    Ts=3000000,
    plot_steps=None,
    member_idx=0  # Miembro del ensemble a graficar
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo de simulación: {checkpoint_path}")

    print(f"Cargando simulación desde: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location='cpu')

    ens_y = data['history_y']  # Shape: [N, M, 3]
    N_sim, M, _ = ens_y.shape
    N = plot_steps if (plot_steps and plot_steps < N_sim) else N_sim

    # Recortar al número de pasos deseado
    ens_y = ens_y[:N]

    # Cargar referencia real Y_true
    if os.path.exists(data_path):
        raw_mat = scipy.io.loadmat(data_path)['u']
        Y_true = raw_mat[3:6, Ts : Ts + N].T  # Shape: [N, 3]
    else:
        raise FileNotFoundError(f"No se encontró el archivo de referencia: {data_path}")

    # Tomar la trayectoria de un miembro del ensemble (ej. el primero)
    # NOTA: Usar el promedio ens_y.mean(dim=1) hace que el atractor colapse
    # por la dispersión caótica. La trayectoria individual refleja la física real.
    y_sim_member = ens_y[:, member_idx, :].float().numpy()

    print("Generando proyecciones 2D del atractor...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]

    for ax, (i, j), (lbl_i, lbl_j) in zip(axes, pairs, labels):
        # 1. Atractor Real
        ax.plot(Y_true[:, i], Y_true[:, j], 
                color='black', alpha=0.4, linewidth=0.5, label='Real Attractor')

        # 2. Atractor Simulado (Lazo Cerrado)
        ax.plot(y_sim_member[:, i], y_sim_member[:, j], 
                color='red', alpha=0.6, linewidth=0.5, label=f'Closed Loop (Member {member_idx})')

        ax.set_xlabel(lbl_i)
        ax.set_ylabel(lbl_j)
        ax.set_title(f'Plano de Fase: {lbl_i} vs {lbl_j}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Closed Loop: Attractor Comparison (Slow Variables Y)', fontsize=14)
    plt.tight_layout()

    os.makedirs('media', exist_ok=True)
    out_path = 'media/closed_loop/attractor_comparison.png'
    plt.savefig(out_path, dpi=300)
    print(f"Gráfico guardado exitosamente en: {out_path}")
    plt.show()

if __name__ == '__main__':
    plot_attractor()