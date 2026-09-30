import torch
import matplotlib.pyplot as plt

def plot_attractor_2d_projections():
    print("Loading simulation history...")
    data = torch.load('checkpoints/simulation_history_100k.pt', map_location='cpu')
    
    history_y = data['history_y'].numpy()  # Shape: [n_steps, n_ensemble, 3]
    Y_true = data['Y_true'].numpy()        # Shape: [n_steps, 3]
    
    #print("Generating 3x5 matrix of 2D projections...")
    # 3 filas (proyecciones), 6 columnas (1 Real + 5 Agentes)
    fig, axes = plt.subplots(3, 5, figsize=(18, 7))
    
    # Variable pairs for phase planes (cada fila será una proyección)
    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]
    
    for row_idx, ((i, j), (lbl_i, lbl_j)) in enumerate(zip(pairs, labels)):
        
        # --- Columna 0: Atractor Real ---
        ax_real = axes[row_idx, 0]
        ax_real.plot(Y_true[:, i], Y_true[:, j], 
                     color='black', alpha=0.6, linewidth=0.6, label='Real')
        ax_real.set_xlabel(lbl_i)
        ax_real.set_ylabel(lbl_j)
        ax_real.set_title(f'Real | {lbl_i} vs {lbl_j}')
        ax_real.grid(True)
        
        # --- Columnas 1 a 4: Muestras del Ensemble (primeros 3 agentes) ---
        for agent_idx in range(4):
            ax_ens = axes[row_idx, agent_idx + 1]
            agent_data = history_y[:, agent_idx, :]  # Trayectoria del agente
            
            ax_ens.plot(agent_data[:, i], agent_data[:, j], 
                        color='red', alpha=0.6, linewidth=0.6, label=f'Agent {agent_idx + 1}')
            ax_ens.set_xlabel(lbl_i)
            ax_ens.set_ylabel(lbl_j)
            ax_ens.set_title(f'Traj {agent_idx + 1} | {lbl_i} vs {lbl_j}')
            ax_ens.grid(True)
            
    #plt.suptitle('Matrix Comparison: Real vs Ensemble Agents by Columns (3x5)', fontsize=16)
    plt.tight_layout()
    plt.savefig('media/closed_loop_attractor_sample.png', dpi=300)
    print("Plots saved to media/closed_loop_attractor_sample.png!")
    plt.show()

if __name__ == '__main__':
    plot_attractor_2d_projections()