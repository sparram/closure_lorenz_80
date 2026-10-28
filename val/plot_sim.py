import torch
import scipy.io
from config import data_path, idx_0, N_train, N_test, K, stride, device
import matplotlib.pyplot as plt

def plot():
    print("Cargando datos de referencia y simulación...")
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    split_idx = idx_0 + N_train
    start_idx = K * stride  # 5000 puntos de historia / calentamiento

    # Cargar simulación
    X_sim = torch.load("cfm_sim.pt", map_location=device)
    sim_steps = X_sim.shape[0] # Debería ser N_test
    print(f"Pasos detectados en la simulación: {sim_steps}")

    # Extraer datos reales correspondientes EXACTAMENTE al tramo simulado
    # Como la simulación arranca en (split_idx + start_idx), la referencia debe arrancar ahí mismo
    X_test = U_raw[0:3][:, split_idx + start_idx : split_idx + start_idx + sim_steps]
    Y_test = U_raw[3:6][:, split_idx + start_idx : split_idx + start_idx + sim_steps]
    Z_test = U_raw[6:9][:, split_idx + start_idx : split_idx + start_idx + sim_steps]

    # Apilar referencias reales en una matriz [9, sim_steps]
    true_data = torch.cat([
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test),
        torch.from_numpy(Z_test)
    ], dim=0)

    # Configuración estética de la figura (3x3 grid)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9
    })

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    var_names = ['$X_1$', '$X_2$', '$X_3$', '$Y_1$', '$Y_2$', '$Y_3$', '$Z_1$', '$Z_2$', '$Z_3$']

    time_axis = range(sim_steps)

    for i in range(9):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Extraer la variable i para todos los ensembles: [sim_steps, N_ens]
        var_ensemble = X_sim[:, :, i]
        
        mean_sim = var_ensemble.mean(dim=1).cpu().numpy()
        std_sim = var_ensemble.std(dim=1).cpu().numpy()
        
        true_val = true_data[i].cpu().numpy()

        # Plotear referencia real y media del ensemble con bandas de confianza
        ax.plot(time_axis, true_val, 'k-', label='True' if i == 0 else "", alpha=0.7, linewidth=1.2)
        ax.plot(time_axis, mean_sim, 'r--', label='CFM Ensemble Mean' if i == 0 else "", linewidth=1.2)
        ax.fill_between(
            time_axis, 
            mean_sim - 2 * std_sim, 
            mean_sim + 2 * std_sim, 
            color='r', alpha=0.2, label='$\pm 2\sigma$ Band' if i == 0 else ""
        )

        ax.set_ylabel(var_names[i])
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True)

    axes[2, 1].set_xlabel('Time Steps')
    plt.suptitle('Closed-Loop Lorenz-80 Hybrid CFM Simulation vs. True Reference', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    #plt.savefig('media/closed_loop/ensemble_validation_3x3.png', dpi=300, bbox_inches='tight')
    print("¡Gráfica guardada exitosamente en 'media/closed_loop/ensemble_validation_3x3.png'!")
    plt.show()

if __name__ == "__main__":
    plot()