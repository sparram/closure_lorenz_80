import os
import random
import numpy as np
import torch
from tqdm import tqdm
from src.load_data import get_dataloaders
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def set_seed(seed=37):
    """ Fijar semillas para reproducibilidad total """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(37)
    
    # Detectar GPU Nvidia (cuda) o Mac Silicon (mps)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Entrenando en dispositivo: {device}")
    torch.set_num_threads(4)
    
    # Subimos batch_size a 1024 para reducir drasticamente el tiempo por epoca
    train_loader, val_loader = get_dataloaders('data/NHLR_data.mat', batch_size=1024)

    model = ConditionalVelocityField().to(device)
    cfm = ConditionalFlowMatcher(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}")

        for y_batch, fast_batch in pbar:
            y_batch, fast_batch = y_batch.to(device), fast_batch.to(device)

            optimizer.zero_grad()
            loss = cfm.compute_loss(y_batch, fast_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_batch, fast_batch in val_loader:
                y_batch, fast_batch = y_batch.to(device), fast_batch.to(device)
                val_loss += cfm.compute_loss(y_batch, fast_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Checkpoint parcial: Guardar solo si es el mejor resultado de validacion
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/cfm_l80.pt')
            print(f" -> Guardado mejor modelo (Val MSE: {val_loss:.6f})")

        # Respaldo por epoca para retomar si la compu se apaga
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, 'checkpoints/last_checkpoint.pt')

if __name__ == '__main__':
    train()