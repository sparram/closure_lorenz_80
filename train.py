import os
import random
import numpy as np
import torch
from tqdm import tqdm
from src.load_data import get_dataloaders
from src.model import ConditionalVelocityField
from src.flow_matching import ConditionalFlowMatcher

def train():
    history_len = 5000
    stride = 50
    k_points = history_len // stride  # 100
    dim_cond = k_points * 3           # 300

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device} | Dim Condición Historia: {dim_cond}")

    train_loader, val_loader = get_dataloaders(
        'data/NN_training_data.mat', 
        batch_size=1024, 
        history_len=history_len, 
        stride=stride
    )

    model = ConditionalVelocityField(dim_cond=dim_cond, hidden_dim=256).to(device)
    cfm = ConditionalFlowMatcher(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}")

        for y_cond_batch, fast_batch in pbar:
            y_cond_batch = y_cond_batch.to(device)
            fast_batch = fast_batch.to(device)

            optimizer.zero_grad()
            loss = cfm.compute_loss(y_cond_batch, fast_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_cond_batch, fast_batch in val_loader:
                y_cond_batch = y_cond_batch.to(device)
                fast_batch = fast_batch.to(device)
                val_loss += cfm.compute_loss(y_cond_batch, fast_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/cfm_model_hlf.pt')
            print(f" -> Mejor modelo guardado (Val MSE: {val_loss:.6f})")

if __name__ == '__main__':
    train()