import os
import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import FNOConditionalVelocityField  
from src.flow_matching import ConditionalFlowMatcher

def run_level2_scatter_validation(
    file_path='data/NN_training_data.mat', 
    stats_path='checkpoints/norm_stats_hlf.pt', 
    model_path='checkpoints/cfm_fno_model_hlf.pt', 
    num_samples=20000,      # Rápido y ligero
    skip_transient=3000000,
    batch_size=1024,       # Lotes eficientes para GPU/CPU
    steps_cfm=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Validación Open Loop FNO - Ligera] Usando dispositivo: {device}")

    # 1. Cargar Estadísticas y Parámetros
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    y_mean, y_std = stats['y_mean'].to(device), stats['y_std'].to(device)
    x_mean, x_std = stats['x_mean'].cpu(), stats['x_std'].cpu()
    z_mean, z_std = stats['z_mean'].cpu(), stats['z_std'].cpu()

    history_len = stats['history_len'] # 5000
    stride = stats['stride']            # 50
    k_points = stats['k_points']        # 100

    # 2. Cargar Datos Reales (tramo necesario)
    raw_data = scipy.io.loadmat(file_path)['u']
    
    start_idx = skip_transient - history_len
    end_idx = skip_transient + num_samples
    slice_data = raw_data[:, start_idx:end_idx]

    X_true = torch.tensor(slice_data[0:3, history_len:].T, dtype=torch.float32)
    Z_true = torch.tensor(slice_data[6:9, history_len:].T, dtype=torch.float32)

    Y_raw = torch.tensor(slice_data[3:6].T, dtype=torch.float32, device=device)
    Y_norm = (Y_raw - y_mean) / y_std

    # 3. Preparación exacta de historiales (Idéntica a load_data.py)
    print(f"Preparando {num_samples} historiales con stride={stride}...")
    Y_cond_list = []
    with torch.no_grad():
        for i in range(num_samples):
            t = i + history_len
            # Extrae la ventana temporal y transpone a forma: (3, k_points) -> (3, 100)
            y_hist = Y_norm[t - history_len : t : stride].T 
            Y_cond_list.append(y_hist)
            
    # Apilamos en un tensor de forma: (num_samples, 3, 100)
    Y_cond_all = torch.stack(Y_cond_list, dim=0)
    
    # 4. Cargar Modelo FNO CFM
    model = FNOConditionalVelocityField(
        dim_fast=6, in_channels=3, modes=16, width=64, time_emb_dim=16
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    # 5. Generar Muestras por Minibatches
    hat_x_list, hat_z_list = [], []
    
    print(f"Muestreando en lotes de {batch_size}...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            cond_batch = Y_cond_all[i : i + batch_size].to(device) # Shape: (batch, 3, 100)
            hx_norm, hz_norm = cfm.sample(cond_batch, steps=steps_cfm)
            hat_x_list.append(hx_norm.cpu())
            hat_z_list.append(hz_norm.cpu())

    hat_x_norm = torch.cat(hat_x_list, dim=0)
    hat_z_norm = torch.cat(hat_z_list, dim=0)

    # 6. Des-normalización
    hat_x = hat_x_norm * x_std + x_mean
    hat_z = hat_z_norm * z_std + z_mean

    # --- GRAFICACIÓN ---
    plot_len = min(100000, num_samples) 
    t_axis = (torch.arange(skip_transient, skip_transient + plot_len) * stats['dt']).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(t_axis, X_true[:plot_len, 0].numpy(), color='black', alpha=0.7, linewidth=1.2, label='Real X1')
    axes[0].plot(t_axis, hat_x[:plot_len, 0].numpy(), color='blue', alpha=0.6, linewidth=1.0, label='Generated X1 (FNO-CFM)')
    axes[0].set_xlabel('Physical Time (t)')
    axes[0].set_ylabel('X1')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(t_axis, Z_true[:plot_len, 0].numpy(), color='black', alpha=0.7, linewidth=1.2, label='Real Z1')
    axes[1].plot(t_axis, hat_z[:plot_len, 0].numpy(), color='blue', alpha=0.6, linewidth=1.0, label='Generated Z1 (FNO-CFM)')
    axes[1].set_xlabel('Physical Time (t)')
    axes[1].set_ylabel('Z1')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Open Loop Validation (Lightweight FNO-CFM)', fontsize=12)
    plt.tight_layout()
    
    os.makedirs('media', exist_ok=True)
    plt.savefig("media/xz_open_loop_fno.png", dpi=200)
    plt.show()
    print("✨ ¡Validación completada con éxito! Gráfica guardada en media/xz_open_loop_fno.png")

if __name__ == '__main__':
    stats = torch.load('checkpoints/norm_stats_hlf.pt')
    test_start_idx = stats['split_idx']  
    print(test_start_idx + 230000)
    
    run_level2_scatter_validation(
        skip_transient=test_start_idx + 100000, 
        num_samples=100500  # Margen extra para los 1500 de ploteo
    )