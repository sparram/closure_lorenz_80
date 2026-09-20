import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def run_level2_scatter_validation(file_path='data/NHLR_data.mat', num_samples=10000, skip_transient=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    raw_data = scipy.io.loadmat(file_path)['u']
    
    # Extraer muestras
    slice_data = raw_data[:, skip_transient:skip_transient+num_samples]
    X_true = torch.tensor(slice_data[0:3].T, dtype=torch.float32)
    Y_true = torch.tensor(slice_data[3:6].T, dtype=torch.float32, device=device)
    Z_true = torch.tensor(slice_data[6:9].T, dtype=torch.float32)
    
    # Cargar modelo
    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80.pt', map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()
    
    with torch.no_grad():
        hat_x, hat_z = cfm.sample(Y_true, steps=50)
        hat_x, hat_z = hat_x.cpu(), hat_z.cpu()
        Y_cpu = Y_true.cpu()

    # --- GRAFICACIÓN SCATTER (Espacio de Fases) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Relación Y1 vs X1 (Real vs Generado)
    axes[0].scatter(Y_cpu[:, 0].numpy(), X_true[:, 0].numpy(), color='black', alpha=0.25, s=8, label='Real (Y1 vs X1)')
    axes[0].scatter(Y_cpu[:, 0].numpy(), hat_x[:, 0].numpy(), color='red', alpha=0.25, s=8, label='Generado CFM (Y1 vs X1)')
    axes[0].set_xlabel('Y1 (Lento)')
    axes[0].set_ylabel('X1 (Rápido)')
    axes[0].set_title('Manifold Condicional: X1 en función de Y1')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Relación Y1 vs Z1 (Real vs Generado)
    axes[1].scatter(Y_cpu[:, 0].numpy(), Z_true[:, 0].numpy(), color='black', alpha=0.25, s=8, label='Real (Y1 vs Z1)')
    axes[1].scatter(Y_cpu[:, 0].numpy(), hat_z[:, 0].numpy(), color='red', alpha=0.25, s=8, label='Generado CFM (Y1 vs Z1)')
    axes[1].set_xlabel('Y1 (Lento)')
    axes[1].set_ylabel('Z1 (Rápido)')
    axes[1].set_title('Manifold Condicional: Z1 en función de Y1')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle('Nivel 2 (Complemento): Estructura del Manifold Condicional')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_level2_scatter_validation()