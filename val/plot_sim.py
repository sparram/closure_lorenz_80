import torch
import scipy.io
from config import data_path, idx_0, N_train, N_test, K, stride, device
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import ot

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
    
    dynamics(U_pred, U_test, split_idx, N_sim)

    X_test = U_raw[0:3][:, split_idx:]
    Y_test = U_raw[3:6][:, split_idx:]
    Z_test = U_raw[6:9][:, split_idx:]
    
    U_test = torch.cat([torch.from_numpy(X_test), torch.from_numpy(Y_test), torch.from_numpy(Z_test)], dim=0)

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
    #plt.show()

def invariant_measure(U_pred, U_test, split_idx, N_sim):
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 13,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    pairs = [(3, 4), (4, 5), (5,3)]
    labels = [('$Y_1$', '$Y_2$'), ('$Y_2$', '$Y_3$'), ('$Y_3$', '$Y_1$')]

    bins = 64
    
    for i in range(3):
        u1_pred = U_pred[:, :, pairs[i][0]]
        u2_pred = U_pred[:, :, pairs[i][1]]
        u1_test = U_test[pairs[i][0]].to(torch.float32)
        u2_test = U_test[pairs[i][1]].to(torch.float32)
        
        u1_pred, u2_pred = u1_pred.flatten(), u2_pred.flatten()
        xmin, xmax = min(u1_pred.min(), u1_test.min()), max(u1_pred.max(), u1_test.max())
        ymin, ymax = min(u2_pred.min(), u2_test.min()), max(u2_pred.max(), u2_test.max())
        x_axis = torch.linspace(xmin, xmax, bins+1)
        y_axis = torch.linspace(ymin, ymax, bins+1)

        x_centers = (x_axis[1:] + x_axis[:-1]) / 2
        y_centers = (y_axis[1:] + y_axis[:-1]) / 2
        X, Y = torch.meshgrid(x_centers, y_centers, indexing='ij')

        H_pred, _ = torch.histogramdd(torch.stack([u1_pred, u2_pred], dim=1), bins=[x_axis, y_axis], density=True)
        H_pred /= H_pred.sum()
        
        H_test, _ = torch.histogramdd(torch.stack([u1_test, u2_test], dim=1), bins=[x_axis, y_axis], density=True)
        H_test /= H_test.sum()
        
        centers_np = torch.stack([X.flatten(), Y.flatten()], axis=1).cpu().numpy()
        M = ot.dist(centers_np, centers_np, metric='euclidean')
        H_pred_np = H_pred.flatten().cpu().numpy()
        H_test_np = H_test.flatten().cpu().numpy()
        w1 = ot.sinkhorn2(H_pred_np, H_test_np, M, reg=0.05)
        w1_str = f"{w1:.2e}"
        w1_base, w1_exp = w1_str.split('e')
        w1_latex = f"{float(w1_base):.2f} \\times 10^{{{int(w1_exp)}}}"

        P_pred = ndimage.gaussian_filter(H_pred, sigma=1.2)
        P_pred /= P_pred.sum()

        P_test = ndimage.gaussian_filter(H_test, sigma=1.2)
        P_test /= P_test.sum()
        
        vmax = max(P_test.max(), P_pred.max())
        
        ax = axes[0,i]
        ax.pcolormesh(X,Y, P_test, cmap='turbo', vmin=0.0, vmax=vmax, shading='gouraud')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.set_title(f"$P(${labels[i][0]}$, ${labels[i][1]}$)$")

        ax = axes[1,i]
        ax.pcolormesh(X,Y, P_pred, cmap='turbo', vmin=0.0, vmax=vmax, shading='gouraud')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.set_title(fr"$P_{{\text{{CFM}}}}(${labels[i][0]}$, ${labels[i][1]}$) - W_1 = {w1_latex}$")
        
    plt.tight_layout()
    plt.savefig("media/cfm_invariant_measure.png", dpi=300, bbox_inches='tight')
    print("Invariant measure successful!")
    #plt.show()
    
if __name__ == "__main__":
    plot()