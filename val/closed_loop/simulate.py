import torch
import scipy.io
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher
from src.physics import rk4_step_y

def run(M=100, N=20000, dt=4.2e-3, Ts=3000000):
    torch.set_num_threads(4)
    torch.manual_seed(37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = torch.load('checkpoints/norm_stats_hlf.pt', map_location=device)
    history_len = stats['history_len'] # 5000
    stride = stats['stride']           # 50
    k_points = stats['k_points']       # 100
    dim_cond = k_points * 3            # 300

    data = scipy.io.loadmat('data/NN_training_data.mat')['u']
    
    # 1. Cargar los 5,000 pasos de historia previa exacta a t=0
    raw_history = torch.from_numpy(data[3:6, Ts - history_len : Ts].T).to(dtype=torch.float32, device=device)
    y_hist_norm = (raw_history - stats['y_mean'].to(device)) / stats['y_std'].to(device) # Shape: (5000, 3)
    
    # Buffer continuo de 5000 pasos para los M miembros: Shape (M, 5000, 3)
    history_buffer = y_hist_norm.unsqueeze(0).repeat(M, 1, 1)

    # Estado inicial Y0 (último punto de la historia + ruido)
    Yn = (y_hist_norm[-1].unsqueeze(0).repeat(M, 1) * stats['y_std'].to(device)) + stats['y_mean'].to(device)
    Yn = Yn + torch.randn(M, 3, device=device) * 1e-2

    model = ConditionalVelocityField(dim_cond=dim_cond, hidden_dim=256).to(device)
    model.load_state_dict(torch.load('checkpoints/cfm_model_hlf.pt', map_location=device, weights_only=True))
    cfm = ConditionalFlowMatcher(model)
    model.eval()

    ens_y = torch.zeros(N, M, 3, dtype=torch.float16, device=device)
    ens_x = torch.zeros(N, M, 3, dtype=torch.float16, device=device)
    ens_z = torch.zeros(N, M, 3, dtype=torch.float16, device=device)

    print(f"[Simulación] Integrando {N} pasos con dt = {dt} y memoria de 5000 puntos...")

    with torch.no_grad():
        for step in range(N):
            # 2. Tomar 1 de cada 50 puntos dentro de los 5,000 pasos y aplanar a (M, 300)
            y_cond_input = history_buffer[:, ::stride, :].reshape(M, -1)

            # 3. Muestrear CFM (X, Z) con el historial de 300 dims
            xi_norm, zi_norm = cfm.sample(y_cond_input, steps=10)

            xi = xi_norm * stats['x_std'].to(device) + stats['x_mean'].to(device)
            zi = zi_norm * stats['z_std'].to(device) + stats['z_mean'].to(device)

            # 4. Promedio del ensemble para acoplamiento RK4
            E_xi = xi.mean(dim=0, keepdim=True).expand(M, 3)
            E_zi = zi.mean(dim=0, keepdim=True).expand(M, 3)

            # 5. Integrar Y un paso dt = 4.2e-3
            Yn = rk4_step_y(Yn, E_xi, E_zi, dt=dt)

            # 6. Actualizar el buffer: desplazar y agregar el nuevo Yn
            yn_norm = (Yn - stats['y_mean'].to(device)) / stats['y_std'].to(device)
            history_buffer[:, :-1, :] = history_buffer[:, 1:, :].clone()
            history_buffer[:, -1, :] = yn_norm  

            # Guardar trayectorias
            ens_x[step] = xi.to(torch.float16)
            ens_y[step] = Yn.to(torch.float16)
            ens_z[step] = zi.to(torch.float16)

            if step % 500 == 0:
                print(f"PROGRESS: {step} / {N} - {100 * step / N} %")

    torch.save({
        'history_x': ens_x.cpu(),
        'history_y': ens_y.cpu(),
        'history_z': ens_z.cpu(),
        'dt': dt
    }, 'checkpoints/simulation_history_memory.pt')

if __name__ == '__main__':
    stats = torch.load('checkpoints/norm_stats_hlf.pt')
    test_start_idx = stats['split_idx']
    
    # Simular en el conjunto de prueba que el modelo NUNCA vio en entrenamiento
    run(M=50, N=100000, dt=4.2e-3, Ts=test_start_idx + 230000)