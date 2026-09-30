import torch
import scipy.io
import matplotlib.pyplot as plt
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

# M : Ensemble size
# N : Number of timesteps
def run(M=100, N=20000, dt=4.2e-2, Ts=3000000):
    torch.set_num_threads(4)
    
    torch.manual_seed(37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    # dt = 4.2e-2 is 10 times greater than the original dt* = 4.2e-3
    # Hence, for the visualization, we take N points every 10 timesteps on the original states
    S = torch.from_numpy(data[0:9, Ts:Ts + 10 * N:10].T).to(dtype=torch.float32, device=device).clone()
    X_true = S[:, 0:3]
    Y_true = S[:, 3:6]
    Z_true = S[:, 6:9]

    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80_nhlr.pt', map_location=device, weights_only=True))
    try:
        print("Trying model compilation")
        model = torch.compile(model)
        print("Model compiled successfully.")
    except (AttributeError, RuntimeError, Exception) as e:
        print(f"The model couldn't be compiled ({e}). We'll use the standard model.")
        
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    # Sample N members of Y (at t=0)
    y0 = Y_true[0].unsqueeze(0)
    Yn = y0.repeat(M, 1) + torch.randn(M, 3, device=device) * 1e-2

    # Arrays for the ensembles
    ens_y = torch.zeros(N, M, 3, dtype=torch.float16, device=device)
    ens_x = torch.zeros(N, M, 3, dtype=torch.float16, device=device)
    ens_z = torch.zeros(N, M, 3, dtype=torch.float16, device=device)
    
    ens_y[0] = Yn.to(torch.float16)

    print(f"[Validación XYZ] Integrando {N} pasos en CPU...")
    stats = torch.load('checkpoints/norm_stats_nhlr.pt', map_location=device)

    with torch.no_grad():
        for step in range(1, N):
            # X_m, Z_m = Call CFM(Y_n)
            y_norm = (Yn - stats['y_mean'].to(device)) / stats['y_std'].to(device)
            xi_norm, zi_norm = cfm.sample(y_norm, steps=5)            
            xi = xi_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            zi = zi_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)

            # Compute E[X_m], E[Z_m] = Xbar, Zbar
            E_xi = xi.mean(dim=0, keepdim=True).expand(M, 3)
            E_zi = zi.mean(dim=0, keepdim=True).expand(M, 3)

            # Y(t=t+1) = F[Y(t), Xbar, Zbar]
            Yn = rk4_step_y(Yn, E_xi, E_zi, dt=dt)

            # Gives us N trajectories of Y, Xbar, Zbar
            # Save ensemble for the actual timestep
            ens_x[step] = xi.to(torch.float16)
            ens_y[step] = Yn.to(torch.float16)
            ens_z[step] = zi.to(torch.float16)
            
            if step % 2000 == 0:
                print(f"  Paso {step}/{N} completado.")
    
    # Save data
    torch.save({
        'history_x': ens_x.cpu(),
        'history_y': ens_y.cpu(),
        'history_z': ens_z.cpu(),
        'X_true': X_true.cpu().to(torch.float16),
        'Y_true': Y_true.cpu().to(torch.float16),
        'Z_true': Z_true.cpu().to(torch.float16),
        'dt': dt
    }, 'checkpoints/simulation_history_100k.pt')

if __name__ == '__main__':
    run()