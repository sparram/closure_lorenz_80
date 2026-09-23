import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

def run_level3_validation_xyz(n_ensemble=50, n_steps=50000, dt=4.2e-3, skip_transient=3000000):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    
    # Extraer referencias reales para X, Y, Z omitiendo el transitorio
    X_true = torch.tensor(raw_data[0:3, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)
    Y_true = torch.tensor(raw_data[3:6, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)
    Z_true = torch.tensor(raw_data[6:9, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)

    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80_nhlr.pt', map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    y_init = Y_true[0].unsqueeze(0)
    y_ensemble = y_init.repeat(n_ensemble, 1) + torch.randn(n_ensemble, 3, device=device) * 1e-2

    history_y = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history_x = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history_z = torch.zeros(n_steps, n_ensemble, 3, device=device)
    
    history_y[0] = y_ensemble

    print(f"[Validación XYZ] Integrando y guardando trayectorias desde el paso {skip_transient}...")
    
    stats = torch.load('checkpoints/norm_stats_nhlr.pt', map_location=device)
    
    with torch.no_grad():
        for step in range(1, n_steps):
            # 1. Obtener la media del ensamble actual (forma: 3,) o (1, 3)
            y_mean = y_ensemble.mean(dim=0)
            
            # 2. DUPLICAR el y_mean 50 veces ANTES de llamar al CFM
            # Esto crea un batch de forma (50, 3) donde todas las filas son el mismo y_mean
            y_single_repeated = y_mean.unsqueeze(0).repeat(n_ensemble, 1)
            
            # 3. Normalizar
            y_norm = (y_single_repeated - stats['y_mean'].to(device)) / stats['y_std'].to(device)
            
            # 4. Ahora cfm.sample ve num_samples = 50. 
            # Generará 50 ruidos aleatorios torch.randn(50, 6) diferentes 
            # para la misma condición y_mean.
            hat_x_norm, hat_z_norm = cfm.sample(y_norm, steps=5)
            
            # Des-normalizar (obtendrás 50 valores distintos de X y Z)
            hat_x = hat_x_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            hat_z = hat_z_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)
            
            # Guardar para las bandas de confianza
            history_x[step] = hat_x
            history_z[step] = hat_z
            
            # 5. Avanzar la física del ensamble
            y_ensemble = rk4_step_y(y_ensemble, hat_x, hat_z, dt=dt)
            history_y[step] = y_ensemble
            
            if step % 500 == 0:
                print(f"  Paso {step}/{n_steps} completado.")

    # --- Process metrics for plotting ---
    t_axis = (torch.arange(n_steps) * dt).cpu().numpy()
    
    # Mean and Standard Deviation for Y
    mean_y = history_y.mean(dim=1).cpu().numpy()
    std_y = history_y.std(dim=1).cpu().numpy()
    y_true_np = Y_true.cpu().numpy()

    # Mean and Standard Deviation for X (ADDED!)
    mean_x = history_x.mean(dim=1).cpu().numpy()
    std_x = history_x.std(dim=1).cpu().numpy() # <-- NEW
    x_true_np = X_true.cpu().numpy()

    # Mean and Standard Deviation for Z (ADDED!)
    mean_z = history_z.mean(dim=1).cpu().numpy()
    std_z = history_z.std(dim=1).cpu().numpy() # <-- NEW
    z_true_np = Z_true.cpu().numpy()

    # --- MULTIPANEL PLOTTING (X, Y, Z) WITH CONFIDENCE BANDS ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Component Y1
    axes[0].plot(t_axis, y_true_np[:, 0], 'k-', label='Real Y1', alpha=0.8)
    axes[0].plot(t_axis, mean_y[:, 0], 'r--', label='Ensemble Mean Y1')
    axes[0].fill_between(t_axis, mean_y[:, 0] - 2 * std_y[:, 0], mean_y[:, 0] + 2 * std_y[:, 0], color='r', alpha=0.2)
    axes[0].set_ylabel('Y1 (Slow)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # 2. Component X1 (Prediction vs Real + Confidence band)
    axes[1].plot(t_axis, x_true_np[:, 0], 'k-', label='Real X1', alpha=0.8)
    axes[1].plot(t_axis, mean_x[:, 0], 'b--', label='Ensemble Mean X1')
    axes[1].fill_between(t_axis, mean_x[:, 0] - 2 * std_x[:, 0], mean_x[:, 0] + 2 * std_x[:, 0], color='b', alpha=0.2) # <-- NEW
    axes[1].set_ylabel('X1 (Fast)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # 3. Component Z1 (Prediction vs Real + Confidence band)
    axes[2].plot(t_axis, z_true_np[:, 0], 'k-', label='Real Z1', alpha=0.8)
    axes[2].plot(t_axis, mean_z[:, 0], 'g--', label='Ensemble Mean Z1')
    axes[2].fill_between(t_axis, mean_z[:, 0] - 2 * std_z[:, 0], mean_z[:, 0] + 2 * std_z[:, 0], color='g', alpha=0.2) # <-- NEW
    axes[2].set_xlabel('Physical time (t)')
    axes[2].set_ylabel('Z1 (Fast)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.suptitle('Closed Loop Validation: Predicted Y (Slow) with X and Z (Fast)')
    plt.tight_layout()
    plt.savefig("media/sim_closed_loop.png")
    plt.show()

if __name__ == '__main__':
    run_level3_validation_xyz()