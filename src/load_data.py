import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_l80_dataset(file_path='data/NHLR_data.mat', t_transient=10.0, dt=4.2e-3):
    data = scipy.io.loadmat(file_path)
    U = data['u']  # Dimensión original: (9, nt)

    # Filtrar Transitorio y separar
    idx_start = int(t_transient / dt)
    Y_data = U[3:6, idx_start:].T
    X = U[0:3, :]
    Z = U[6:9, :]
    XZ_data = np.vstack([X, Z])[:, idx_start:].T

    # Convertir a tensores de PyTorch
    y_tensor = torch.tensor(Y_data, dtype=torch.float32)
    xz_tensor = torch.tensor(XZ_data, dtype=torch.float32)

    return TensorDataset(y_tensor, xz_tensor)


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

# Test unitario de dimensiones
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders('../data/NN_training_data.mat', batch_size=256)
    for y_batch, xz_batch in train_loader:
        # It should be
        # y_batch shape: torch.Size([256, 3])
        # xz_batch shape: torch.Size([256, 6])
        print(y_batch.shape)
        print(xz_batch.shape)
        break