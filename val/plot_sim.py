import torch
import scipy.io
from config import data_path, idx_0, N_train, N_test, K, stride, device
import matplotlib.pyplot as plt

def plot():
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    split_idx = 5030476
    X_sim = torch.load("cfm_sim.pt", map_location=device, weights_only=True)
    N_sim = X_sim.shape[0]
    
    X_test = U_raw[0:3][:, split_idx : split_idx + N_sim]
    Y_test = U_raw[3:6][:, split_idx : split_idx + N_sim]
    Z_test = U_raw[6:9][:, split_idx : split_idx + N_sim]
    
    U_test = torch.cat([torch.from_numpy(X_test), torch.from_numpy(Y_test), torch.from_numpy(Z_test)], dim=0)
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9
    })

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    labels = ['$X_1$', '$X_2$', '$X_3$', '$Y_1$', '$Y_2$', '$Y_3$', '$Z_1$', '$Z_2$', '$Z_3$']

    time = torch.Tensor(range(split_idx, split_idx + N_sim)) * 45 / 86400

    for i in range(9):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        u_real = U_test[i].cpu().numpy()
        u_pred = X_sim[:, :, i]
        
        u_mean = u_pred.mean(dim=1).cpu().numpy()
        u_std = u_pred.std(dim=1).cpu().numpy()
        lw_limit = u_mean - 2 * u_std
        up_limit = u_mean + 2 * u_std
        
        ax.plot(time, u_real, 'k-', label='Data' if i == 0 else "", alpha=0.7, linewidth=1.2)
        ax.plot(time, u_mean, 'r--', label='Mean' if i == 0 else "", linewidth=1.2)
        ax.fill_between(time, lw_limit, up_limit, color='r', alpha=0.2, label='$\pm 2\sigma$ Band' if i == 0 else "")
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True)

    axes[2, 1].set_xlabel('Time (Days)')
    #plt.suptitle('A Posteriori CFM-Predicted Dynamics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    #plt.savefig('media/closed_loop/ensemble_validation_3x3.png', dpi=300, bbox_inches='tight')
    print("Plotting successful!")
    plt.show()

if __name__ == "__main__":
    plot()