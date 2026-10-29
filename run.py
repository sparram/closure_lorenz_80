import torch
import scipy.io
import math
from src.cfm import CFM, FlowModel
from config import data_path, idx_0, N_train, N_test, N_ens, sigma0, K, stride, device

torch.manual_seed(37)

# PHYSICAL SYSTEM PARAMETERS
a1, a2, a3 = 1.0, 1.0, 3.0
v0 = 1.0 / 48.0
b1 = (a1 - a2 - a3) / 2.0
b2 = (a2 - a3 - a1) / 2.0
b3 = (a3 - a1 - a2) / 2.0
c = math.sqrt(b1 * b2 + b2 * b3 + b3 * b1)

def dYdt(x, y, z):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]

    dy1dt = (- a3 * b3 * x2 * y3 - a2 * b2 * y2 * x3 + c * (a3 - a2) * y2 * y3 - a1 * x1 - v0 * a1**2 * y1) / a1
    dy2dt = (- a1 * b1 * x3 * y1 - a3 * b3 * y3 * x1 + c * (a1 - a3) * y3 * y1 - a2 * x2 - v0 * a2**2 * y2) / a2
    dy3dt = (- a2 * b2 * x1 * y2 - a1 * b1 * y1 * x2 + c * (a2 - a1) * y1 * y2 - a3 * x3 - v0 * a3**2 * y3) / a3

    return torch.stack([dy1dt, dy2dt, dy3dt], dim=-1)

def rk4_step_y(y, x, z, dt=4.2e-3):
    k1 = dYdt(x, y, z)
    k2 = dYdt(x, y + 0.5 * dt * k1, z)
    k3 = dYdt(x, y + 0.5 * dt * k2, z)
    k4 = dYdt(x, y + dt * k3, z)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run():
    torch.set_num_threads(4)
    
    stats = torch.load("norm_stats_hlf.pt", map_location=device, weights_only=True)
    meanX, stdX = stats["x_mean"].to(device, dtype=torch.float32), stats["x_std"].to(device, dtype=torch.float32)
    meanY, stdY = stats["y_mean"].to(device, dtype=torch.float32), stats["y_std"].to(device, dtype=torch.float32)
    meanZ, stdZ = stats["z_mean"].to(device, dtype=torch.float32), stats["z_std"].to(device, dtype=torch.float32)
    
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    test_start_idx = stats['split_idx']
    split_idx = test_start_idx + 230000 # We traslate the test idx 230000 units to capture a more interesting part of the attractor
    
    N_hist = K * stride
    histY = torch.from_numpy(U_raw[3:6, split_idx - N_hist : split_idx].T)
    histY = histY.to(dtype=torch.float32, device=device)
    
    histY_norm = (histY - meanY) / stdY    
    hist_buffer = histY_norm.unsqueeze(0).repeat(N_ens, 1, 1)

    torch.manual_seed(37)
    Y0 = (histY_norm[-1].unsqueeze(0) * stdY) + meanY
    Y_pred = Y0.repeat(N_ens, 1) + sigma0 * torch.randn(N_ens, 3, device=device)

    v_flow = FlowModel().to(device)
    weights = torch.load("cfm_model_hlf.pt", map_location=device, weights_only=True)
    v_flow.load_state_dict(weights)
    cfm = CFM(v_flow)
    v_flow.eval()

    ensX = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)
    ensY = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)
    ensZ = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)

    with torch.no_grad():
        for step in range(N_test):
            Y_cond = hist_buffer[:, ::stride, :].reshape(N_ens, 3 * K)
            
            CFM_pred = cfm.sample(Y_cond)
            X_pred_norm, Z_pred_norm = CFM_pred
            X_pred = X_pred_norm * stdX + meanX
            Z_pred = Z_pred_norm * stdZ + meanZ

            Y_pred = rk4_step_y(Y_pred, X_pred, Z_pred)

            Y_pred_norm = (Y_pred - meanY) / stdY
            hist_buffer[:, :-1, :] = hist_buffer[:, 1:, :].clone()
            hist_buffer[:, -1, :] = Y_pred_norm  

            ensX[step] = X_pred
            ensY[step] = Y_pred
            ensZ[step] = Z_pred

            if step % 500 == 0:
                print(f"PROGRESS: {step} / {N_test} ({100 * step / N_test:.1f}%)")
        
    X_sim = torch.cat([ensX, ensY, ensZ], dim=-1)
    torch.save(X_sim, "cfm_sim.pt")
    print("A POSTERIORI SIMULATION COMPLETED!")

if __name__ == "__main__":
    run()