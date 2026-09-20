import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

def run_level3_validation_xyz(n_ensemble=50, n_steps=10000, dt=4.2e-3, skip_transient=4000000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    
    # Extraer referencias reales para X, Y, Z omitiendo el transitorio
    X_true = torch.tensor(raw_data[0:3, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)
    Y_true = torch.tensor(raw_data[3:6, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)
    Z_true = torch.tensor(raw_data[6:9, skip_transient:skip_transient + n_steps].T, dtype=torch.float32, device=device)

    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80.pt', map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    y_init = Y_true[0].unsqueeze(0)
    y_ensemble = y_init.repeat(n_ensemble, 1) + torch.randn(n_ensemble, 3, device=device) * 1e-2

    history_y = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history_x = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history_z = torch.zeros(n_steps, n_ensemble, 3, device=device)
    
    history_y[0] = y_ensemble

    print(f"[Validación XYZ] Integrando y guardando trayectorias desde el paso {skip_transient}...")
    
    stats = torch.load('checkpoints/norm_stats.pt', map_location=device)
    
    with torch.no_grad():
        for step in range(1, n_steps):
            y_norm = (y_ensemble - stats['y_mean'].to(device)) / stats['y_std'].to(device)
            
            # Puedes probar con steps=20 o steps=3 según lo que desees comparar
            hat_x_norm, hat_z_norm = cfm.sample(y_norm, steps=3)
            
            hat_x = hat_x_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            hat_z = hat_z_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)
            
            # Guardar predicciones de este paso
            history_x[step] = hat_x
            history_z[step] = hat_z
            
            y_ensemble = rk4_step_y(y_ensemble, hat_x, hat_z, dt=dt)
            history_y[step] = y_ensemble
            
            if step % 500 == 0:
                print(f"  Paso {step}/{n_steps} completado.")

    # Procesar métricas para graficar
    t_axis = (torch.arange(n_steps) * dt).cpu().numpy()
    
    mean_y = history_y.mean(dim=1).cpu().numpy()
    std_y = history_y.std(dim=1).cpu().numpy()
    y_true_np = Y_true.cpu().numpy()

    mean_x = history_x.mean(dim=1).cpu().numpy()
    x_true_np = X_true.cpu().numpy()

    mean_z = history_z.mean(dim=1).cpu().numpy()
    z_true_np = Z_true.cpu().numpy()

    # --- GRAFICACIÓN MULTIPANEL (X, Y, Z) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Componente Y1
    axes[0].plot(t_axis, y_true_np[:, 0], 'k-', label='Real Y1', alpha=0.8)
    axes[0].plot(t_axis, mean_y[:, 0], 'r--', label='Ensemble Mean Y1')
    axes[0].fill_between(t_axis, mean_y[:, 0] - 2 * std_y[:, 0], mean_y[:, 0] + 2 * std_y[:, 0], color='r', alpha=0.2)
    axes[0].set_ylabel('Y1 (Lento)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # 2. Componente X1 (Predicción vs Real)
    axes[1].plot(t_axis, x_true_np[:, 0], 'k-', label='Real X1', alpha=0.8)
    axes[1].plot(t_axis, mean_x[:, 0], 'b--', label='CFM Pred X1')
    axes[1].set_ylabel('X1 (Rápido)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # 3. Componente Z1 (Predicción vs Real)
    axes[2].plot(t_axis, z_true_np[:, 0], 'k-', label='Real Z1', alpha=0.8)
    axes[2].plot(t_axis, mean_z[:, 0], 'g--', label='CFM Pred Z1')
    axes[2].set_xlabel('Tiempo físico (t)')
    axes[2].set_ylabel('Z1 (Rápido)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.suptitle('Validación Cruzada de Variables: Y (Lento) junto a X y Z (Rápidos)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_level3_validation_xyz()