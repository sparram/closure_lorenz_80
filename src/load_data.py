import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HistoryDataset(Dataset):
    def __init__(self, file_path='data/NN_training_data.mat', t_transient=10.0, dt=4.2e-3, 
                 history_len=5000, stride=50, split='train', train_ratio=0.8,
                 save_stats=True, stats_path='checkpoints/norm_stats_hlf.pt'):
        
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

        total_points = len(Y_raw)
        split_idx = int(total_points * train_ratio)

        # 1. Calcular estadísticas ÚNICAMENTE con los datos de entrenamiento (para evitar data leakage)
        y_train = Y_raw[:split_idx]
        xz_train = XZ_raw[:split_idx]

        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std = torch.clamp(y_train.std(dim=0, keepdim=True), min=1e-7)

        xz_mean = xz_train.mean(dim=0, keepdim=True)
        xz_std = torch.clamp(xz_train.std(dim=0, keepdim=True), min=1e-7)

        if save_stats and split == 'train':
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            stats = {
                'y_mean': y_mean, 'y_std': y_std,
                'xz_mean': xz_mean, 'xz_std': xz_std,
                'x_mean': xz_mean[:, :3], 'x_std': xz_std[:, :3],
                'z_mean': xz_mean[:, 3:], 'z_std': xz_std[:, 3:],
                'history_len': history_len,
                'stride': stride,
                'dt': dt,
                'k_points': history_len // stride,
                'split_idx': split_idx + idx_start  # Índice absoluto en la matriz original 'u'
            }
            torch.save(stats, stats_path)
            print(f"[load_data] Estadísticas guardadas. Punto de corte Test (split_idx): {stats['split_idx']}")

        # 2. Cortar el dataset según el split solicitado
        if split == 'train':
            self.Y_norm = (Y_raw[:split_idx] - y_mean) / y_std
            self.XZ_norm = (XZ_raw[:split_idx] - xz_mean) / xz_std
        elif split == 'val' or split == 'test':
            # Incluimos los history_len puntos anteriores al corte para no perder el inicio del test set
            self.Y_norm = (Y_raw[split_idx - history_len:] - y_mean) / y_std
            self.XZ_norm = (XZ_raw[split_idx - history_len:] - xz_mean) / xz_std

        self.history_len = history_len
        self.stride = stride

    def __len__(self):
        return len(self.Y_norm) - self.history_len

    def __getitem__(self, idx):
        t = idx + self.history_len
        y_hist = self.Y_norm[t - self.history_len : t : self.stride]
        y_cond = y_hist.reshape(-1)
        target_xz = self.XZ_norm[t]
        return y_cond, target_xz


def get_dataloaders(file_path='data/NN_training_data.mat', batch_size=1024, 
                    history_len=5000, stride=50, train_ratio=0.8):
    
    train_dataset = HistoryDataset(file_path=file_path, history_len=history_len, stride=stride, split='train', train_ratio=train_ratio)
    val_dataset = HistoryDataset(file_path=file_path, history_len=history_len, stride=stride, split='val', train_ratio=train_ratio, save_stats=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader