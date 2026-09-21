import torch
import numpy as np
from src.model import ConditionalVelocityField

def export_weights_to_jax():
    # 1. Definir el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instanciar la arquitectura del modelo
    model = ConditionalVelocityField().to(device)

    # 3. Cargar los pesos que ya entrenaste
    print("Cargando el checkpoint de PyTorch...")
    checkpoint_path = 'checkpoints/cfm_l80_nhlr.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Extraer los tensores, pasarlos a CPU, convertirlos a NumPy y limpiar los nombres
    weights_dict = {}
    for name, param in model.state_dict().items():
        # JAX prefiere nombres sin puntos para las claves de diccionarios
        clean_name = name.replace('.', '_')
        weights_dict[clean_name] = param.detach().cpu().numpy()

    # 5. Guardar el archivo comprimido .npz
    output_path = 'checkpoints/cfm_weights.npz'
    np.savez(output_path, **weights_dict)
    print(f"¡Pesos exportados exitosamente en '{output_path}' para usarlos en JAX!")

if __name__ == '__main__':
    export_weights_to_jax()