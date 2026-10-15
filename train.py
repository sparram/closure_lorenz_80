import os
import torch
from tqdm import tqdm
from src.load_data import get_dataloaders
from src.model import FNOConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def train():
    history_len = 5000
    stride = 50
    k_points = history_len // stride  # 100 puntos en el dominio temporal de la historia

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device} | Arquitectura: FNO 1D (Modos: 16, Ancho: 64)")

    # Obtener dataloaders con la estructura matricial (3, k_points)
    train_loader, val_loader = get_dataloaders(
        'data/NN_training_data.mat', 
        batch_size=1024, 
        history_len=history_len, 
        stride=stride
    )

    # Inicializar el campo de velocidades basado en FNO
    model = FNOConditionalVelocityField(
        dim_fast=6, 
        in_channels=3, 
        modes=16,          # Modos de Fourier retenidos
        width=64,          # Canales ocultos internos del operador
        time_emb_dim=16
    ).to(device)

    cfm = ConditionalFlowMatcher(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}")

        for y_hist_batch, fast_batch in pbar:
            y_hist_batch = y_hist_batch.to(device) # Shape: (batch, 3, k_points)
            fast_batch = fast_batch.to(device)     # Shape: (batch, 6)

            optimizer.zero_grad()
            loss = cfm.compute_loss(y_hist_batch, fast_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Fase de Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_hist_batch, fast_batch in val_loader:
                y_hist_batch = y_hist_batch.to(device)
                fast_batch = fast_batch.to(device)
                val_loss += cfm.compute_loss(y_hist_batch, fast_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f" -> Fin Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Guardar el mejor modelo basado en la pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/cfm_fno_model_hlf.pt')
            print(f"    ✨ ¡Mejor modelo guardado en checkpoints/cfm_fno_model_hlf.pt!")

if __name__ == '__main__':
    train()