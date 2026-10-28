import torch
from src.cfm import CFM, FlowModel
from config import data_path, idx_0, N_train, N_test, N_ens, sigma0, K, stride, device
import scipy.io
import math

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
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    split_idx = idx_0+N_train
    X_test = U_raw[0:3][:, split_idx : split_idx + N_test]
    Y_test = U_raw[3:6][:, split_idx : split_idx + N_test]
    Z_test = U_raw[6:9][:, split_idx : split_idx + N_test]

    stats = torch.load("norm_stats.pt", map_location=device, weights_only=True)
    meanX, stdX = stats["meanX"], stats["stdX"]
    meanY, stdY = stats["meanY"], stats["stdY"]
    meanZ, stdZ = stats["meanZ"], stats["stdZ"]

    print("="*100)
    Y_hist = U_raw[3:6][:, split_idx - K * stride : split_idx : stride]
    print(Y_hist.shape)
    Y_hist = torch.from_numpy(Y_hist).t()
    print(Y_hist.shape)
    
    Y_hist = (Y_hist - meanY) / stdY
    print(Y_hist.shape)
    Y_hist = Y_hist.repeat(N_ens, 1, 1)
    print(Y_hist.shape)
    print("="*100)
    
    Y0 = torch.from_numpy(Y_test)[:,0][:, None]
    Y_pred = Y0 + sigma0 * torch.randn((3,N_ens))
    print(Y_pred.shape)
        
    v_flow = FlowModel()
    weights = torch.load("cfm_model.pt", map_location=device, weights_only=True)
    v_flow.load_state_dict(weights)
    cfm = CFM(v_flow)
    
    X_sim = []
    for k in range(N_test):
        X_in_test = Y_hist.reshape(N_ens, 3*K)
        #print(Y_hist.shape, X_in_test.shape)
        X_in_test = X_in_test.to(dtype=torch.float32)
        
        X_out_test = cfm.sample(X_in_test)
        X_pred, Z_pred = X_out_test
        
        X_pred = X_pred.t() * stdX[:, None] + meanX[:, None]
        Z_pred = Z_pred.t() * stdZ[:, None] + meanZ[:, None]

        Y_pred = rk4_step_y(Y_pred.t(), X_pred.t(), Z_pred.t()).t()

        #print("FORMA:", X_pred.shape, Y_pred.shape, Z_pred.shape)
        X_sim.append(torch.vstack([X_pred, Y_pred, Z_pred]))
        #print(torch.vstack([X_pred, Y_pred, Z_pred]).shape)

        Y_pred_norm = (Y_pred - meanY[:, None]) / stdY[:, None]
        Y_hist[:, :-1, :] = Y_hist[:, 1:, :].clone()
        Y_hist[:, -1, :] = Y_pred_norm.t()
    
        if k % 500 == 0:
            print(f"PROGRESS: {k} / {N_test}")
        
    X_sim = torch.stack(X_sim)
    print(X_sim.shape)
    torch.save(X_sim, "cfm_sim.pt")
    
if __name__ == "__main__":
    run()