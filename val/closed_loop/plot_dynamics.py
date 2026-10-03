import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

# M : Ensemble size
# N : Number of timesteps
def plot_dynamics(M=100, N=200000, dt=4.2e-2, Ts=3000000):

    print("Loading simulation history...")
    data = torch.load('checkpoints/simulation_history_100k.pt', map_location='cpu')
    
    ens_x = data['history_x']  # Shape: [n_steps, n_ensemble, 3]
    ens_y = data['history_y']  # Shape: [n_steps, n_ensemble, 3]
    ens_z = data['history_z']  # Shape: [n_steps, n_ensemble, 3]
    X_true = data['X_true'].numpy()    # Shape: [n_steps, 3]
    Y_true = data['Y_true'].numpy()    # Shape: [n_steps, 3]
    Z_true = data['Z_true'].numpy()    # Shape: [n_steps, 3]
    
    # --- Process metrics for plotting ---
    t_axis = (torch.arange(N) * dt).cpu().numpy()
    mean_y = ens_y.mean(dim=1).cpu().numpy()
    std_y = ens_y.std(dim=1).cpu().numpy()
    y_true_np = Y_true

    mean_x = ens_x.mean(dim=1).cpu().numpy()
    std_x = ens_x.std(dim=1).cpu().numpy()
    x_true_np = X_true

    mean_z = ens_z.mean(dim=1).cpu().numpy()
    std_z = ens_z.std(dim=1).cpu().numpy()
    z_true_np = Z_true

    # --- MULTIPANEL PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_axis, y_true_np[:, 0], 'k-', label='Real Y1', alpha=0.8)
    axes[0].plot(t_axis, mean_y[:, 0], 'r--', label='Ensemble Mean Y1')
    axes[0].fill_between(t_axis, mean_y[:, 0] - 2 * std_y[:, 0], mean_y[:, 0] + 2 * std_y[:, 0], color='r', alpha=0.2)
    axes[0].set_ylabel('Y1 (Slow)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].plot(t_axis, x_true_np[:, 0], 'k-', label='Real X1', alpha=0.8)
    axes[1].plot(t_axis, mean_x[:, 0], 'b--', label='Ensemble Mean X1')
    axes[1].fill_between(t_axis, mean_x[:, 0] - 2 * std_x[:, 0], mean_x[:, 0] + 2 * std_x[:, 0], color='b', alpha=0.2)
    axes[1].set_ylabel('X1 (Fast)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    axes[2].plot(t_axis, z_true_np[:, 0], 'k-', label='Real Z1', alpha=0.8)
    axes[2].plot(t_axis, mean_z[:, 0], 'g--', label='Ensemble Mean Z1')
    axes[2].fill_between(t_axis, mean_z[:, 0] - 2 * std_z[:, 0], mean_z[:, 0] + 2 * std_z[:, 0], color='g', alpha=0.2)
    axes[2].set_xlabel('Physical time (t)')
    axes[2].set_ylabel('Z1 (Fast)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.suptitle('Closed Loop Validation: Predicted Y (Slow) with X and Z (Fast)')
    plt.tight_layout()
    plt.savefig("media/sim_closed_loop.png")
    plt.show()

if __name__ == '__main__':
    plot_dynamics()