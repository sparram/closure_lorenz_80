import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

def run_level3_simulation(n_ensemble=50, n_steps=5000, dt=4.2e-3, skip_transient=20000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    
    # Extraer Y_true omitiendo el periodo transitorio
    Y_true = torch.tensor(
        raw_data[3:6, skip_transient:skip_transient + n_steps].T, 
        dtype=torch.float32, 
        device=device
    )

    model = ConditionalVelocityField().to(device)
    model.load_state_dict(
        torch.load('checkpoints/cfm_l80.pt', map_location=device, weights_only=True)
    )
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    # Punto inicial correspondiente a t_0 = skip_transient * dt en el atractor
    y_init = Y_true[0].unsqueeze(0)
    y_ensemble = y_init.repeat(n_ensemble, 1) + torch.randn(n_ensemble, 3, device=device) * 1e-3

    history = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history[0] = y_ensemble

    print(f"[Nivel 3 PyTorch] Integrando ensemble online desde el paso {skip_transient} (t = {skip_transient * dt:.2f})...")
    
    # Cargar modelo y estadísticas de normalización
    stats = torch.load('checkpoints/norm_stats.pt', map_location=device)
    
    with torch.no_grad():
        for step in range(1, n_steps):
            # 1. Normalizar Y antes de pasar a la red
            y_norm = (y_ensemble - stats['y_mean'].to(device)) / stats['y_std'].to(device)
    
            # 2. Generar muestras normalizadas desde el CFM
            hat_x_norm, hat_z_norm = cfm.sample(y_norm, steps=30)
    
            # 3. Des-normalizar a espacio físico real
            hat_x = hat_x_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            hat_z = hat_z_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)
    
            # 4. Integrar en RK4 físico
            y_ensemble = rk4_step_y(y_ensemble, hat_x, hat_z, dt=dt)
            history[step] = y_ensemble
            
            if step % 500 == 0:
                print(f"  Paso {step}/{n_steps} completado.")

    mean_traj = history.mean(dim=1).cpu()
    std_traj = history.std(dim=1).cpu()
    t_axis = (torch.arange(n_steps) * dt).numpy()
    y_true_cpu = Y_true.cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(t_axis, y_true_cpu[:, 0], 'k-', label='Referencia Real (Y1)', alpha=0.8)
    plt.plot(t_axis, mean_traj[:, 0].numpy(), 'r--', label='Media del Ensemble')
    plt.fill_between(
        t_axis, 
        (mean_traj[:, 0] - 2 * std_traj[:, 0]).numpy(), 
        (mean_traj[:, 0] + 2 * std_traj[:, 0]).numpy(), 
        color='r', alpha=0.25, label='Incertidumbre Ensemble (±2σ)'
    )
    plt.ylim(-3.0, 3.0)
    plt.xlabel('Tiempo físico relativo (t)')
    plt.ylabel('y1')
    plt.title(f'Nivel 3: Cierre Neuronal L80 Online (Inicio en t_0 = {skip_transient * dt:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_level3_simulation()