import torch
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.cfm import CFM, FlowModel
from src.data_loader import MemoryDataset
from config import data_path, idx_0, N_train, N_test, K, stride, device

def evaluate_open_loop(batch_size=512, plot_steps=10000):
    print("Cargando estadísticas de normalización y modelo...")
    stats = torch.load("norm_stats.pt", map_location=device, weights_only=True)
    meanX, stdX = stats["meanX"].to(device), stats["stdX"].to(device)
    meanY, stdY = stats["meanY"].to(device), stats["stdY"].to(device)
    meanZ, stdZ = stats["meanZ"].to(device), stats["stdZ"].to(device)

    # Cargar el modelo entrenado
    v_flow = FlowModel().to(device)
    weights = torch.load("cfm_model.pt", map_location=device, weights_only=True)
    v_flow.load_state_dict(weights)
    cfm = CFM(v_flow)
    v_flow.eval()

    # Usar el dataset de prueba y envolverlo en un DataLoader con batch_size grande
    test_dataset = MemoryDataset(split="test", K=K, stride=stride)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Generando predicciones open-loop vectorizadas (batch_size={batch_size})...")

    pred_x_list = []
    pred_z_list = []
    true_x_list = []
    true_z_list = []

    total_processed = 0

    with torch.no_grad():
        for input_batch, out_batch in test_loader:
            if total_processed >= plot_steps:
                break
                
            # input_batch shape: [B, 300], out_batch shape: [B, 6]
            cond = input_batch.to(device, dtype=torch.float32)
            
            # Muestrear el CFM de forma masiva (paralelizada en GPU)
            x_hat_norm, z_hat_norm = cfm.sample(cond, N_steps=10)
            
            # Desnormalizar predicciones y verdaderos de golpe
            x_hat = x_hat_norm * stdX + meanX
            z_hat = z_hat_norm * stdZ + meanZ
            
            true_x = out_batch[:, 0:3].to(device) * stdX + meanX
            true_z = out_batch[:, 3:6].to(device) * stdZ + meanZ

            pred_x_list.append(x_hat.cpu())
            pred_z_list.append(z_hat.cpu())
            true_x_list.append(true_x.cpu())
            true_z_list.append(true_z.cpu())
            
            total_processed += cond.shape[0]

    # Concatenar todos los lotes eficientemente
    pred_x = torch.cat(pred_x_list, dim=0)[:plot_steps].numpy()
    pred_z = torch.cat(pred_z_list, dim=0)[:plot_steps].numpy()
    true_x = torch.cat(true_x_list, dim=0)[:plot_steps].numpy()
    true_z = torch.cat(true_z_list, dim=0)[:plot_steps].numpy()

    # --- PLOTEO DE RESULTADOS OPEN-LOOP ---
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10
    })

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)

    # Variables X (X1, X2, X3)
    for idx in range(3):
        axes[idx, 0].plot(true_x[:, idx], 'k-', label='True $X$', alpha=0.8, linewidth=1.2)
        axes[idx, 0].plot(pred_x[:, idx], 'r--', label='CFM Open-Loop', alpha=0.8, linewidth=1.2)
        axes[idx, 0].set_ylabel(f'$X_{idx+1}$')
        axes[idx, 0].grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            axes[idx, 0].legend(loc='upper right', frameon=True)
            axes[idx, 0].set_title('Fast Variables ($X$): Regression Quality')

    # Variables Z (Z1, Z2, Z3)
    for idx in range(3):
        axes[idx, 1].plot(true_z[:, idx], 'k-', label='True $Z$', alpha=0.8, linewidth=1.2)
        axes[idx, 1].plot(pred_z[:, idx], 'b--', label='CFM Open-Loop', alpha=0.8, linewidth=1.2)
        axes[idx, 1].set_ylabel(f'$Z_{idx+1}$')
        axes[idx, 1].grid(True, linestyle='--', alpha=0.4)
        if idx == 0:
            axes[idx, 1].legend(loc='upper right', frameon=True)
            axes[idx, 1].set_title('Fast Variables ($Z$): Regression Quality')

    axes[2, 0].set_xlabel('Time Steps')
    axes[2, 1].set_xlabel('Time Steps')

    plt.tight_layout()
    #plt.savefig('media/closed_loop/open_loop_sanity_check.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica open-loop guardada con éxito en 'media/closed_loop/open_loop_sanity_check.png'!")
    plt.show()

if __name__ == "__main__":
    evaluate_open_loop(batch_size=1024, plot_steps=500000)