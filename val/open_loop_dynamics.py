import os
import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def run_level2_scatter_validation(
    file_path='data/NN_training_data.mat', 
    stats_path='checkpoints/norm_stats_hlf.pt', 
    model_path='checkpoints/cfm_model_hlf.pt',
    num_samples=100000, 
    skip_transient=3000000,
    batch_size=5000,
    steps_cfm=20
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Validación Open Loop] Usando dispositivo: {device}")

    # 1. Cargar Estadísticas y Parámetros de Historia
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    y_mean, y_std = stats['y_mean'].to(device), stats['y_std'].to(device)
    x_mean, x_std = stats['x_mean'].cpu(), stats['x_std'].cpu()
    z_mean, z_std = stats['z_mean'].cpu(), stats['z_std'].cpu()

    history_len = stats['history_len'] # 5000
    stride = stats['stride']           # 50
    k_points = stats['k_points']       # 100
    dim_cond = k_points * 3            # 300

    # 2. Cargar Datos Reales con margen para el historial inicial
    raw_data = scipy.io.loadmat(file_path)['u']
    
    start_idx = skip_transient - history_len
    end_idx = skip_transient + num_samples
    slice_data = raw_data[:, start_idx:end_idx]

    # Extraer variables objetivo (num_samples,) desfasadas por history_len
    X_true = torch.tensor(slice_data[0:3, history_len:].T, dtype=torch.float32)
    Z_true = torch.tensor(slice_data[6:9, history_len:].T, dtype=torch.float32)

    # Tensor Y continuo (history_len + num_samples, 3)
    Y_raw = torch.tensor(slice_data[3:6].T, dtype=torch.float32, device=device)
    Y_norm = (Y_raw - y_mean) / y_std

    # 3. Vectorización del Historial de 300 Dims con .unfold()
    # Y_norm.unfold(0, history_len, 1) genera ventanas deslizantes de tamaño 5000
    unfolded_y = Y_norm.unfold(0, history_len, 1)[:num_samples]     # Shape: (num_samples, 3, 5000)
    subsampled_y = unfolded_y[:, :, ::stride]                       # Shape: (num_samples, 3, 100)
    Y_cond_all = subsampled_y.transpose(1, 2).reshape(num_samples, -1) # Shape: (num_samples, 300)

    # 4. Cargar Modelo CFM
    model = ConditionalVelocityField(dim_cond=dim_cond, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    # 5. Generar Muestras por Minibatches
    hat_x_list, hat_z_list = [], []
    
    print(f"Muestreando {num_samples} puntos en lotes de {batch_size}...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            cond_batch = Y_cond_all[i : i + batch_size].to(device)
            hx_norm, hz_norm = cfm.sample(cond_batch, steps=steps_cfm)
            hat_x_list.append(hx_norm.cpu())
            hat_z_list.append(hz_norm.cpu())

    hat_x_norm = torch.cat(hat_x_list, dim=0)
    hat_z_norm = torch.cat(hat_z_list, dim=0)

    # 6. Des-normalización
    hat_x = hat_x_norm * x_std + x_mean
    hat_z = hat_z_norm * z_std + z_mean

    # --- GRAFICACIÓN (Primeros 2,000 pasos para visualización clara) ---
    plot_len = min(100000, num_samples)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Relación X1
    axes[0].plot(X_true[:plot_len, 0].numpy()[::100], color='black', alpha=0.6, linewidth=1.2, label='Real X1')
    axes[0].plot(hat_x[:plot_len, 0].numpy()[::100], color='red', alpha=0.6, linewidth=1.0, label='Generated X1')
    axes[0].set_xlabel('Pasos de tiempo')
    axes[0].set_ylabel('X1')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Relación Z1
    axes[1].plot(Z_true[:plot_len, 0].numpy()[::100], color='black', alpha=0.6, linewidth=1.2, label='Real Z1')
    axes[1].plot(hat_z[:plot_len, 0].numpy()[::100], color='red', alpha=0.6, linewidth=1.0, label='Generated Z1')
    axes[1].set_xlabel('Pasos de tiempo')
    axes[1].set_ylabel('Z1')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Open Loop Validation: Real Fast variables vs Approximated via CFM (History Condition)')
    plt.tight_layout()
    
    os.makedirs('media', exist_ok=True)
    plt.savefig("media/xz_open_loop.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    run_level2_scatter_validation()