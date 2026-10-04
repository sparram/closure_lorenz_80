import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class HistoryDataset(Dataset):
    def __init__(self, file_path='data/NN_training_data.mat', t_transient=10.0, dt=4.2e-3, 
                 history_len=5000, stride=50, save_stats=True, stats_path='checkpoints/norm_stats_hlf.pt'):
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {file_path}")

        data = scipy.io.loadmat(file_path)
        U = data['u']  # Shape original: (9, nt)

        idx_start = int(t_transient / dt)
        Y_raw = U[3:6, idx_start:].T
        X_raw = U[0:3, idx_start:].T
        Z_raw = U[6:9, idx_start:].T
        
        XZ_raw = torch.tensor(np.hstack([X_raw, Z_raw]), dtype=torch.float32)
        Y_raw = torch.tensor(Y_raw, dtype=torch.float32)

        # Estadísticas de normalización Z-score
        y_mean = Y_raw.mean(dim=0, keepdim=True)
        y_std = torch.clamp(Y_raw.std(dim=0, keepdim=True), min=1e-7)

        xz_mean = XZ_raw.mean(dim=0, keepdim=True)
        xz_std = torch.clamp(XZ_raw.std(dim=0, keepdim=True), min=1e-7)

        if save_stats:
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            stats = {
                'y_mean': y_mean, 'y_std': y_std,
                'xz_mean': xz_mean, 'xz_std': xz_std,
                'x_mean': xz_mean[:, :3], 'x_std': xz_std[:, :3],
                'z_mean': xz_mean[:, 3:], 'z_std': xz_std[:, 3:],
                'history_len': history_len,
                'stride': stride,
                'dt': dt,
                'k_points': history_len // stride
            }
            torch.save(stats, stats_path)

        self.Y_norm = (Y_raw - y_mean) / y_std
        self.XZ_norm = (XZ_raw - xz_mean) / xz_std

        self.history_len = history_len
        self.stride = stride

    def __len__(self):
        return len(self.Y_norm) - self.history_len

    def __getitem__(self, idx):
        t = idx + self.history_len
        
        # Extraer 100 puntos espaciados cada 50 pasos dentro de los 5,000 pasos de historia
        y_hist = self.Y_norm[t - self.history_len : t : self.stride] # Shape: (100, 3)
        y_cond = y_hist.reshape(-1)                                  # Shape: (300,)
        target_xz = self.XZ_norm[t]

        return y_cond, target_xz


def get_dataloaders(file_path='data/NN_training_data.mat', batch_size=1024, 
                    history_len=5000, stride=50, val_split=0.2, seed=42):
    
    dataset = HistoryDataset(file_path=file_path, history_len=history_len, stride=stride)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader