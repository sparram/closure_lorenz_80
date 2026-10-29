import torch
import numpy as np

# --- IMPORTA SEGÚN LA CARPETA ---
# Si estás en la carpeta del Enfoque 1:
# from src.model import ConditionalVelocityField
# from src.flow_matching import ConditionalFlowMatcher

# Si estás en la carpeta l80_fast (Enfoque 2):
from src.cfm import FlowModel, CFM

def test_batch_parity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Fijar semilla estrictamente
    torch.manual_seed(37)
    np.random.seed(37)

    # 2. Definir lote de M = 5 muestras
    M = 5  
    dim_cond = 300
    
    dummy_cond = torch.randn(M, dim_cond, device=device)
    
    # 3. Inicializar modelo
    # (Usa ConditionalVelocityField si estás en el enfoque 1, o FlowModel en el enfoque 2)
    model = FlowModel(out_dim=6, cond_dim=dim_cond, time_emb_dim=16, hidden_dim=256).to(device)
    
    # 4. Cargar pesos
    weights_path = 'cfm_model_hlf.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    cfm = CFM(model)
    model.eval()

    # 5. Volver a fijar semilla justo antes del muestreo estocástico (z_tau inicial)
    torch.manual_seed(37)
    with torch.no_grad():
        xi_norm, zi_norm = cfm.sample(dummy_cond, N_steps=10)
        
    print(f"Muestreo completado para M={M} muestras.")
    print("\n--- Resultados xi_norm (M=5, 3 dims) ---")
    print(xi_norm)
    print("\n--- Resultados zi_norm (M=5, 3 dims) ---")
    print(zi_norm)
    
    # Guardar para comparación cruzada
    torch.save({'xi': xi_norm.cpu(), 'zi': zi_norm.cpu()}, 'batch_output.pt')
    print("\nResultados guardados en 'batch_output.pt'")

if __name__ == '__main__':
    test_batch_parity()