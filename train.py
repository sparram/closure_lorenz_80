import os
import torch
from src.load_data import get_dataloaders
from src.models import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando entrenamiento en dispositivo: {device}")

    # 1. Cargar datos
    train_loader, val_loader = get_dataloaders('data/NN_training_data.mat', batch_size=256)

    # 2. Inicializar arquitectura y trainer
    model = ConditionalVelocityField().to(device)
    cfm = ConditionalFlowMatcher(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    os.makedirs('checkpoints', exist_ok=True)

    # 3. Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for y_batch, fast_batch in train_loader:
            y_batch, fast_batch = y_batch.to(device), fast_batch.to(device)

            loss = cfm.compute_loss(y_batch, fast_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_batch, fast_batch in val_loader:
                y_batch, fast_batch = y_batch.to(device), fast_batch.to(device)
                val_loss += cfm.compute_loss(y_batch, fast_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    # Guardar pesos
    torch.save(model.state_dict(), 'checkpoints/cfm_l80.pt')
    print("Entrenamiento completado. Modelo guardado en 'checkpoints/cfm_l80.pt'.")

if __name__ == '__main__':
    train()