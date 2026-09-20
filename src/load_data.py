import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_l80_dataset(file_path='data/NN_training_data.mat', t_transient=10.0, dt=4.2e-3, save_stats=True, stats_path='checkpoints/norm_stats.pt'):
    data = scipy.io.loadmat(file_path)
    U = data['u']  # Dimensión original: (9, nt)

    # Filtrar Transitorio
    idx_start = int(t_transient / dt)
    Y_data = U[3:6, idx_start:].T
    X = U[0:3, :]
    Z = U[6:9, :]
    XZ_data = np.vstack([X, Z])[:, idx_start:].T

    # Convertir a tensores de PyTorch
    y_tensor = torch.tensor(Y_data, dtype=torch.float32)
    xz_tensor = torch.tensor(XZ_data, dtype=torch.float32)

    # 1. Calcular media y desviación estándar
    y_mean = y_tensor.mean(dim=0, keepdim=True)
    y_std = y_tensor.std(dim=0, keepdim=True)
    
    xz_mean = xz_tensor.mean(dim=0, keepdim=True)
    xz_std = xz_tensor.std(dim=0, keepdim=True)

    # Evitar división por cero
    y_std = torch.clamp(y_std, min=1e-7)
    xz_std = torch.clamp(xz_std, min=1e-7)

    # 2. Guardar estadísticas para usar en inferencia (Nivel 2 y 3)
    if save_stats:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats = {
            'y_mean': y_mean, 'y_std': y_std,
            'xz_mean': xz_mean, 'xz_std': xz_std,
            'x_mean': xz_mean[:, :3], 'x_std': xz_std[:, :3],
            'z_mean': xz_mean[:, 3:], 'z_std': xz_std[:, 3:]
        }
        torch.save(stats, stats_path)
        print(f"[load_data] Estadísticas de normalización guardadas en: {stats_path}")

    # 3. Aplicar normalización Z-score
    y_norm = (y_tensor - y_mean) / y_std
    xz_norm = (xz_tensor - xz_mean) / xz_std

    return TensorDataset(y_norm, xz_norm)


def get_dataloaders(file_path='data/NN_training_data.mat', batch_size=256, val_split=0.1, dt=4.2e-3):
    full_dataset = load_l80_dataset(file_path=file_path, dt=dt)

    # Split de validación
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Crear iteradores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders('data/NN_training_data.mat', batch_size=256)
    for y_batch, xz_batch in train_loader:
        print("y_batch normalizado (media ~0, std ~1):", y_batch.mean().item(), y_batch.std().item())
        print("xz_batch normalizado (media ~0, std ~1):", xz_batch.mean().item(), xz_batch.std().item())
        break