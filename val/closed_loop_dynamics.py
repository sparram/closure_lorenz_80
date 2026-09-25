import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

# M : Ensemble size
# N : Number of timesteps
def closed_loop_dynamics(M=50, N=100000, dt=4.2e-3, Ts=3000000):
    torch.set_num_threads(4)
    
    torch.manual_seed(37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    
    X_true = torch.tensor(data[0:3, Ts:Ts + N].T, dtype=torch.float32, device=device)
    Y_true = torch.tensor(data[3:6, Ts:Ts + N].T, dtype=torch.float32, device=device)
    Z_true = torch.tensor(data[6:9, Ts:Ts + N].T, dtype=torch.float32, device=device)

    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80_nhlr.pt', map_location=device, weights_only=True))
    try:
        print("Trying model compilation")
        model = torch.compile(model)
        print("Model compiled successfully.")
    except (AttributeError, RuntimeError, Exception) as e:
        print(f"The model couldn't be compiled ({e}). We'll use the standard model.")
        
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    # Sample N members of Y (at t=0)
    y0 = Y_true[0].unsqueeze(0)
    Yn = y0.repeat(M, 1) + torch.randn(M, 3, device=device) * 1e-2

    # Arrays for the ensembles
    ens_y = torch.zeros(N, M, 3, device=device)
    ens_x = torch.zeros(N, M, 3, device=device)
    ens_z = torch.zeros(N, M, 3, device=device)
    
    ens_y[0] = Yn

    print(f"[Validación XYZ] Integrando {N} pasos en CPU...")
    stats = torch.load('checkpoints/norm_stats_nhlr.pt', map_location=device)

    with torch.no_grad():
        for step in range(1, N):
            # X_m, Z_m = Call CFM(Y_n)
            y_norm = (Yn - stats['y_mean'].to(device)) / stats['y_std'].to(device)
            xi_norm, zi_norm = cfm.sample(y_norm, steps=5)            
            xi = xi_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            zi = zi_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)

            # Compute E[X_m], E[Z_m] = Xbar, Zbar
            E_xi = xi.mean(dim=0, keepdim=True).expand(M, 3)
            E_zi = zi.mean(dim=0, keepdim=True).expand(M, 3)

            # Y(t=t+1) = F[Y(t), Xbar, Zbar]
            Yn = rk4_step_y(Yn, E_xi, E_zi, dt=dt)

            # Gives us N trajectories of Y, Xbar, Zbar
            # Save ensemble for the actual timestep
            ens_x[step] = xi
            ens_y[step] = Yn
            ens_z[step] = zi
            
            if step % 2000 == 0:
                print(f"  Paso {step}/{N} completado.")

    return
    pass
    
    # --- GUARDAR TRAYECTORIAS PARA USO FUTURO ---
    print("Saving ensemble data...")
    torch.save({
        'history_x': ens_x.cpu(),
        'history_y': ens_y.cpu(),
        'history_z': ens_z.cpu(),
        'X_true': X_true.cpu(),
        'Y_true': Y_true.cpu(),
        'Z_true': Z_true.cpu(),
        'dt': dt
    }, 'checkpoints/simulation_history_100k.pt')

    # --- Process metrics for plotting ---
    t_axis = (torch.arange(N) * dt).cpu().numpy()
    mean_y = ens_y.mean(dim=1).cpu().numpy()
    std_y = ens_y.std(dim=1).cpu().numpy()
    y_true_np = Y_true.cpu().numpy()

    mean_x = ens_x.mean(dim=1).cpu().numpy()
    std_x = ens_x.std(dim=1).cpu().numpy()
    x_true_np = X_true.cpu().numpy()

    mean_z = ens_z.mean(dim=1).cpu().numpy()
    std_z = ens_z.std(dim=1).cpu().numpy()
    z_true_np = Z_true.cpu().numpy()

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
    closed_loop_dynamics()