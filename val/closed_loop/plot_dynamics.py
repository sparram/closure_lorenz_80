import os
import torch
import scipy.io
import matplotlib.pyplot as plt

def plot_dynamics(
    checkpoint_path='checkpoints/simulation_history_memory.pt',
    data_path='data/NN_training_data.mat',
    Ts=3000000,
    plot_steps=None  # Permite limitar el número de pasos a graficar (ej. 20000)
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo de simulación: {checkpoint_path}")

    print(f"Cargando simulación desde: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location='cpu')

    ens_x = data['history_x']  # Shape: [N, M, 3]
    ens_y = data['history_y']  # Shape: [N, M, 3]
    ens_z = data['history_z']  # Shape: [N, M, 3]
    dt = data.get('dt', 4.2e-3)

    N_sim, M, _ = ens_y.shape
    N = plot_steps if (plot_steps and plot_steps < N_sim) else N_sim

    # Recortar al número de pasos solicitados
    ens_x = ens_x[:N]
    ens_y = ens_y[:N]
    ens_z = ens_z[:N]

    # Cargar datos reales de referencia partiendo del mismo instante Ts
    if os.path.exists(data_path):
        raw_mat = scipy.io.loadmat(data_path)['u']
        X_true = raw_mat[0:3, Ts : Ts + N].T
        Y_true = raw_mat[3:6, Ts : Ts + N].T
        Z_true = raw_mat[6:9, Ts : Ts + N].T
    else:
        raise FileNotFoundError(f"No se encontró el archivo de datos de referencia: {data_path}")

    # Eje de tiempo físico
    t_axis = (torch.arange(N) * dt).numpy()

    # Promedio y desviación estándar del ensemble
    mean_y = ens_y.mean(dim=1).float().numpy()
    std_y = ens_y.std(dim=1).float().numpy()

    mean_x = ens_x.mean(dim=1).float().numpy()
    std_x = ens_x.std(dim=1).float().numpy()

    mean_z = ens_z.mean(dim=1).float().numpy()
    std_z = ens_z.std(dim=1).float().numpy()

    # --- GRAFICACIÓN MULTIPANEL ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Variable Lenta Y1
    axes[0].plot(t_axis, Y_true[:, 0], 'k-', label='Real Y1', alpha=0.8)
    axes[0].plot(t_axis, mean_y[:, 0], 'r--', label='Ensemble Mean Y1')
    axes[0].fill_between(t_axis, mean_y[:, 0] - 2 * std_y[:, 0], mean_y[:, 0] + 2 * std_y[:, 0], color='r', alpha=0.2)
    axes[0].set_ylabel('Y1 (Lenta)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Variable Rápida X1
    axes[1].plot(t_axis, X_true[:, 0], 'k-', label='Real X1', alpha=0.8)
    axes[1].plot(t_axis, mean_x[:, 0], 'b--', label='Ensemble Mean X1')
    axes[1].fill_between(t_axis, mean_x[:, 0] - 2 * std_x[:, 0], mean_x[:, 0] + 2 * std_x[:, 0], color='b', alpha=0.2)
    axes[1].set_ylabel('X1 (Rápida)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Variable Rápida Z1
    axes[2].plot(t_axis, Z_true[:, 0], 'k-', label='Real Z1', alpha=0.8)
    axes[2].plot(t_axis, mean_z[:, 0], 'g--', label='Ensemble Mean Z1')
    axes[2].fill_between(t_axis, mean_z[:, 0] - 2 * std_z[:, 0], mean_z[:, 0] + 2 * std_z[:, 0], color='g', alpha=0.2)
    axes[2].set_xlabel('Tiempo físico (t)')
    axes[2].set_ylabel('Z1 (Rápida)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Closed Loop Validation: Predicted Y (Slow) with X and Z (Fast)')
    plt.tight_layout()

    os.makedirs('media/closed_loop', exist_ok=True)
    plt.savefig("media/closed_loop/dynamics_comparison.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    stats = torch.load('checkpoints/norm_stats_hlf.pt')
    test_start_idx = stats['split_idx']
    plot_dynamics(Ts=test_start_idx + 230000)