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
    
    # 1. Cargar estadísticas de normalización
    stats = torch.load("norm_stats_hlf.pt", map_location=device, weights_only=True)
    meanX, stdX = stats["x_mean"].to(device, dtype=torch.float32), stats["x_std"].to(device, dtype=torch.float32)
    meanY, stdY = stats["y_mean"].to(device, dtype=torch.float32), stats["y_std"].to(device, dtype=torch.float32)
    meanZ, stdZ = stats["z_mean"].to(device, dtype=torch.float32), stats["z_std"].to(device, dtype=torch.float32)
    
    # 2. Cargar datos desde MATLAB
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]

    # --- CAMBIO CLAVE AQUÍ PARA EMParejar el índice de inicio ---
    test_start_idx = stats['split_idx'] # Esto lee el punto exacto de separación de entrenamiento
    split_idx = test_start_idx + 230000  # Coincide exactamente con el primer script
    
    N_hist = K * stride  # 5000 puntos de historial total

    print(f"[Simulación] Cargando {N_hist} pasos de historia previa a t=0 desde el índice {split_idx}...")
    
    # Extraer los N_hist pasos previos exactos para inicializar el buffer
    raw_history = torch.from_numpy(U_raw[3:6, split_idx - N_hist : split_idx].T).to(dtype=torch.float32, device=device)
    y_hist_norm = (raw_history - meanY) / stdY  # Forma: [N_hist, 3]
    
    # Buffer continuo de historial para los N_ens miembros: [N_ens, N_hist, 3]
    history_buffer = y_hist_norm.unsqueeze(0).repeat(N_ens, 1, 1)

    # Estado inicial Y0 (último punto de la historia en escala real + ruido estocástico)
    Y0_real = (y_hist_norm[-1].unsqueeze(0) * stdY) + meanY
    
    # Fijar la semilla justo antes del torch.randn para asegurar que el ruido inicial sea idéntico
    torch.manual_seed(37)
    Y_pred = Y0_real.repeat(N_ens, 1) + sigma0 * torch.randn(N_ens, 3, device=device)

    # 3. Cargar el modelo CFM entrenado
    v_flow = FlowModel().to(device)
    weights = torch.load("cfm_model_hlf.pt", map_location=device, weights_only=True)
    v_flow.load_state_dict(weights)
    cfm = CFM(v_flow)
    v_flow.eval()

    # Tensores para almacenar la simulación completa
    ens_x = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)
    ens_y = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)
    ens_z = torch.zeros(N_test, N_ens, 3, dtype=torch.float32, device=device)

    print(f"[Simulación] Integrando {N_test} pasos en bucle cerrado con N_ens = {N_ens}...")

    with torch.no_grad():
        for step in range(N_test):
            # A. Extraer submuestreo con el stride y aplanar a [N_ens, 3*K]
            y_cond_input = history_buffer[:, ::stride, :].reshape(N_ens, 3 * K)
            
            # B. Muestrear CFM (X, Z normalizados)
            X_out_test = cfm.sample(y_cond_input)
            X_pred_norm, Z_pred_norm = X_out_test

            # C. Desnormalizar a escala física real para el RK4
            X_pred = X_pred_norm * stdX + meanX
            Z_pred = Z_pred_norm * stdZ + meanZ

            # D. Integración física con RK4
            Y_pred = rk4_step_y(Y_pred, X_pred, Z_pred)

            # E. Actualizar el buffer deslizante: desplazar y agregar el nuevo Y normalizado
            yn_norm = (Y_pred - meanY) / stdY
            history_buffer[:, :-1, :] = history_buffer[:, 1:, :].clone()
            history_buffer[:, -1, :] = yn_norm  

            # Guardar en tensores temporales
            ens_x[step] = X_pred
            ens_y[step] = Y_pred
            ens_z[step] = Z_pred

            if step < 3: # Imprimir solo los primeros 3 pasos
                print(f"--- PASO {step} ---")
                print("Y_pred medio:", Y_pred.mean(dim=0))
                print("X_pred primero:", X_pred[0, :3])

            if step % 500 == 0:
                print(f"PROGRESS: {step} / {N_test} ({100 * step / N_test:.1f}%)")
        
    # Apilar todo en un único tensor final de forma [N_test, N_ens, 9] (compatible con tu plot_sim.py)
    X_sim = torch.stack([ens_x, ens_y, ens_z], dim=2) # O torch.cat según prefieras el orden de columnas
    # Nota: Si tu plot_sim.py espera concatenar X, Y, Z por columnas [N_test, N_ens, 9]:
    X_sim = torch.cat([ens_x, ens_y, ens_z], dim=-1)

    print("Forma final del tensor guardado:", X_sim.shape)
    torch.save(X_sim, "cfm_sim.pt")
    print("¡Simulación completada y guardada con éxito en 'cfm_sim.pt'!")

if __name__ == "__main__":
    run()