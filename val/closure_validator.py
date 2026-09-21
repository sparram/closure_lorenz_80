import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def run_level2_scatter_validation(file_path='data/NHLR_data.mat', stats_path='checkpoints/norm_stats.pt', num_samples=200000, skip_transient=4000000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar Estadísticas de Normalización
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    y_mean, y_std = stats['y_mean'].to(device), stats['y_std'].to(device)
    x_mean, x_std = stats['x_mean'].cpu(), stats['x_std'].cpu()
    z_mean, z_std = stats['z_mean'].cpu(), stats['z_std'].cpu()

    # 2. Cargar Datos Reales
    raw_data = scipy.io.loadmat(file_path)['u']
    slice_data = raw_data[:, skip_transient:skip_transient+num_samples]
    
    X_true = torch.tensor(slice_data[0:3].T, dtype=torch.float32)
    Y_raw = torch.tensor(slice_data[3:6].T, dtype=torch.float32, device=device)
    Z_true = torch.tensor(slice_data[6:9].T, dtype=torch.float32)

    # Normalizar Y para dárselo como condición al modelo
    Y_norm = (Y_raw - y_mean) / y_std

    # 3. Cargar Modelo y Generar Muestras Normalizadas
    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80_best.pt', map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    with torch.no_grad():
        hat_x_norm, hat_z_norm = cfm.sample(Y_norm, steps=3)
        hat_x_norm, hat_z_norm = hat_x_norm.cpu(), hat_z_norm.cpu()

    # 4. Des-normalizar las Muestras Generadas a Unidades Físicas
    hat_x = hat_x_norm * x_std + x_mean
    hat_z = hat_z_norm * z_std + z_mean
    Y_cpu = Y_raw.cpu()

    # --- GRAFICACIÓN SCATTER (Espacio de Fases) ---
    fig, axes = plt.subplots(figsize=(14, 12))
    axes.plot(hat_z[:, 0].numpy(), hat_x[:, 0].numpy(), color='red', alpha=0.25, label='Generated CFM')
    axes.plot(Z_true[:, 0].numpy(), X_true[:, 0].numpy(), color='black', label='Real')
    axes.set_xlabel('Z1')
    axes.set_ylabel('X1')
    axes.set_title('Conditional Manifold: X1 vs Z1')
    axes.legend()
    axes.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_level2_scatter_validation()