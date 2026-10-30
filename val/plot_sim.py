import torch
import scipy.io
from config import data_path, idx_0, N_train, N_test, K, stride, device
import matplotlib.pyplot as plt

def plot():
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    split_idx = 5030476
    U_pred = torch.load("cfm_sim.pt", map_location=device, weights_only=True)
    N_sim = U_pred.shape[0]
    
    X_test = U_raw[0:3][:, split_idx : split_idx + N_sim]
    Y_test = U_raw[3:6][:, split_idx : split_idx + N_sim]
    Z_test = U_raw[6:9][:, split_idx : split_idx + N_sim]
    
    U_test = torch.cat([torch.from_numpy(X_test), torch.from_numpy(Y_test), torch.from_numpy(Z_test)], dim=0)
    print(U_test.shape, U_pred.shape)
    
    #dynamics(U_pred, U_test, split_idx, N_sim)
    invariant_measure(U_pred, U_test, split_idx, N_sim)
    

def dynamics(U_pred, U_test, split_idx, N_sim):
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 13,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    labels = ['$Y_1$', '$X_1$', '$Z_1$']
    colors = ['r', 'b', 'g']
    order = [3, 0, 6]
    time = torch.Tensor(range(split_idx, split_idx + N_sim)) * 45 / 86400

    for i in range(3):
        row = i
        ax = axes[row]

        idx = order[i] 
        u_real = U_test[idx].cpu().numpy()
        u_pred = U_pred[:, :, idx]
        
        u_mean = u_pred.mean(dim=1).cpu().numpy()
        u_std = u_pred.std(dim=1).cpu().numpy()
        lw_limit = u_mean - 2 * u_std
        up_limit = u_mean + 2 * u_std

        ax.plot(time, u_real, '-', color='k', label='Data', alpha=0.9, linewidth=1.2)
        ax.plot(time, u_mean, '--', color=colors[i], label='CFM Mean', linewidth=1.2)
        ax.fill_between(time, lw_limit, up_limit, color=colors[i], alpha=0.2)
        ax.set_ylabel(labels[i])
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, linestyle='-', alpha=0.4)

    axes[2].set_xlabel('Time (Days)')
    plt.tight_layout()
    plt.savefig('media/cfm_dynamics.png', dpi=300, bbox_inches='tight')
    print("Plotting successful!")
    plt.show()

def invariant_measure(U_pred, U_test, split_idx, N_sim):
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 13,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    pairs = [(3, 4), (4, 5), (5,3)]
    labels = [('$Y_1$', '$Y_2$'), ('$Y_2$', '$Y_3$'), ('$Y_3$', '$Y_1$')]

    bins = 80
    
    for i in range(3):
        u1 = U_pred[:, :, pairs[i][0]]
        u2 = U_pred[:, :, pairs[i][1]]
        u1, u2 = u1.flatten(), u2.flatten()
        x_axis = torch.linspace(u1.min(), u1.max(), bins+1)
        y_axis = torch.linspace(u2.min(), u2.max(), bins+1)

        x_centers = (x_axis[1:] + x_axis[:-1]) / 2
        y_centers = (y_axis[1:] + y_axis[:-1]) / 2
        X, Y = torch.meshgrid(x_centers, y_centers, indexing='ij')

        print(torch.stack([u1, u2], dim=1).shape)
        H_pred, _ = torch.histogramdd(torch.stack([u1, u2], dim=1), bins=[x_axis, y_axis], density=True)
        H_pred /= H_pred.sum()

        ax = axes[i]
        print(X.shape, Y.shape, H_pred.shape)
        ax.pcolormesh(X,Y, H_pred, cmap='turbo', vmin=0.0, vmax=H_pred.max(), shading='gouraud')
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    plot()