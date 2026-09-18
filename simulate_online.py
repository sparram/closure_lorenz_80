import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from src.models import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

# Parámetros físicos en PyTorch
a1, a2, a3 = 1.0, 1.0, 3.0
v0 = 1.0 / 48.0
b1 = (a1 - a2 - a3) / 2.0
b2 = (a2 - a3 - a1) / 2.0
b3 = (a3 - a1 - a2) / 2.0
c = np.sqrt(b1 * b2 + b2 * b3 + b3 * b1)

def dYdt_torch(y, x, z):
    """ Ecuación diferencial tensorial dy/dt en PyTorch """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]

    dy1dt = (- a3 * b3 * x2 * y3 - a2 * b2 * y2 * x3 + c * (a3 - a2) * y2 * y3 - a1 * x1 - v0 * a1**2 * y1) / a1
    dy2dt = (- a1 * b1 * x3 * y1 - a3 * b3 * y3 * x1 + c * (a1 - a3) * y3 * y1 - a2 * x2 - v0 * a2**2 * y2) / a2
    dy3dt = (- a2 * b2 * x1 * y2 - a1 * b1 * y1 * x2 + c * (a2 - a1) * y1 * y2 - a3 * x3 - v0 * a3**2 * y3) / a3

    return torch.stack([dy1dt, dy2dt, dy3dt], dim=-1)

def rk4_online_step(cfm, y_ensemble, dt):
    """ Paso RK4 acoplado con muestreo estocástico de Flow Matching """
    hat_x, hat_z = cfm.sample(y_ensemble, steps=5)

    k1 = dYdt_torch(y_ensemble, hat_x, hat_z)
    k2 = dYdt_torch(y_ensemble + 0.5 * dt * k1, hat_x, hat_z)
    k3 = dYdt_torch(y_ensemble + 0.5 * dt * k2, hat_x, hat_z)
    k4 = dYdt_torch(y_ensemble + dt * k3, hat_x, hat_z)

    return y_ensemble + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def run_simulation(n_ensemble=100, n_steps=20000, dt=4.2e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Cargar trayectoria real de referencia
    data = scipy.io.loadmat('data/NN_training_data.mat')
    Y_true = data['u'][3:6, :n_steps].T

    # 2. Cargar modelo CFM
    model = ConditionalVelocityField().to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_l80.pt', map_location=device))
    cfm = ConditionalFlowMatcher(model)

    # 3. Inicializar ensemble N x 3
    y_init = torch.tensor(Y_true[0], dtype=torch.float32, device=device).unsqueeze(0)
    y_ensemble = y_init.repeat(n_ensemble, 1) + torch.randn(n_ensemble, 3, device=device) * 1e-3

    history = torch.zeros(n_steps, n_ensemble, 3, device=device)
    history[0] = y_ensemble

    # 4. Bucle de integración online
    print(f"Simulando ensemble de {n_ensemble} miembros durante {n_steps} pasos...")
    for step in range(1, n_steps):
        y_ensemble = rk4_online_step(cfm, y_ensemble, dt)
        history[step] = y_ensemble

    # 5. Visualización de estadísticas del ensemble
    history = history.cpu().numpy()
    mean_traj = history.mean(axis=1)
    std_traj = history.std(axis=1)
    t_axis = np.arange(n_steps) * dt

    plt.figure(figsize=(12, 5))
    plt.plot(t_axis, Y_true[:, 0], 'k-', label='Referencia Real (Y1)', alpha=0.8)
    plt.plot(t_axis, mean_traj[:, 0], 'r--', label='Media del Ensemble')
    plt.fill_between(t_axis, 
                     mean_traj[:, 0] - 2*std_traj[:, 0], 
                     mean_traj[:, 0] + 2*std_traj[:, 0], 
                     color='r', alpha=0.25, label='Incertidumbre ±2σ')
    plt.xlabel('Tiempo físico (t)')
    plt.ylabel('y1')
    plt.title('Cierre Neuronal L80 via Conditional Flow Matching (Monte Carlo Ensemble)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_simulation()