import os
import torch
import scipy.io
import matplotlib.pyplot as plt

def plot_attractor_2d_projections(
    checkpoint_path='checkpoints/simulation_history_memory.pt',
    data_path='data/NN_training_data.mat',
    Ts=3000000,
    plot_steps=None,
    num_agents=4  # Número de miembros del ensemble a graficar
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo de simulación: {checkpoint_path}")

    print(f"Cargando simulación desde: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location='cpu')

    ens_y = data['history_y']  # Shape: [N_sim, M, 3]
    N_sim, M, _ = ens_y.shape
    N = plot_steps if (plot_steps and plot_steps < N_sim) else N_sim

    # Validar que tengamos suficientes miembros en el ensemble
    if M < num_agents:
        print(f"Advertencia: M ({M}) es menor que num_agents ({num_agents}). Ajustando num_agents a {M}.")
        num_agents = M

    # Recortar al número de pasos deseado
    ens_y = ens_y[:N]

    # Cargar referencia real Y_true
    if os.path.exists(data_path):
        raw_mat = scipy.io.loadmat(data_path)['u']
        Y_true = raw_mat[3:6, Ts : Ts + N].T  # Shape: [N, 3]
    else:
        raise FileNotFoundError(f"No se encontró el archivo de referencia: {data_path}")

    # Matriz de subplots: 3 filas (proyecciones) x (1 Real + num_agents Columnas)
    fig, axes = plt.subplots(3, 1 + num_agents, figsize=(3.5 * (1 + num_agents), 7), sharex='row', sharey='row')

    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]

    for row_idx, ((i, j), (lbl_i, lbl_j)) in enumerate(zip(pairs, labels)):
        
        # --- Columna 0: Atractor Real ---
        ax_real = axes[row_idx, 0]
        ax_real.plot(Y_true[:, i], Y_true[:, j], 
                     color='black', alpha=0.5, linewidth=0.5, label='Real')
        ax_real.set_xlabel(lbl_i)
        ax_real.set_ylabel(lbl_j)
        ax_real.set_title(f'Real | {lbl_i} vs {lbl_j}')
        ax_real.grid(True, linestyle='--', alpha=0.5)

        # --- Columnas 1 a num_agents: Agentes del Ensemble ---
        for agent_idx in range(num_agents):
            ax_ens = axes[row_idx, agent_idx + 1]
            agent_data = ens_y[:, agent_idx, :].float().numpy()

            ax_ens.plot(agent_data[:, i], agent_data[:, j], 
                        color='red', alpha=0.5, linewidth=0.5, label=f'Agent {agent_idx + 1}')
            ax_ens.set_xlabel(lbl_i)
            ax_ens.set_ylabel(lbl_j)
            ax_ens.set_title(f'Agente {agent_idx + 1} | {lbl_i} vs {lbl_j}')
            ax_ens.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Matrix Comparison: Real Attractor vs Ensemble Agents', fontsize=14)
    plt.tight_layout()

    os.makedirs('media', exist_ok=True)
    out_path = 'media/closed_loop/attractor_sample.png'
    plt.savefig(out_path, dpi=300)
    print(f"Gráficos guardados exitosamente en: {out_path}")
    plt.show()

if __name__ == '__main__':
    plot_attractor_2d_projections()