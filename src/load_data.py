import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HistoryDataset(Dataset):
    def __init__(self, file_path='data/NN_training_data.mat', T0=10.0, dt=4.2e-3, 
                 N_hist=5000, stride=50, split='train', train_ratio=0.8,
                 save_stats=True, stats_path='checkpoints/norm_stats_hlf.pt'):

        # 0. Check if data path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data Path was not found at: {file_path}")

        # 1. Load the data (X, Y, Z L80 simulations)
        data = scipy.io.loadmat(file_path)
        U = data['u']  

        # We skip the simulation transient time
        idx_t0 = int(T0 / dt)
        Y_raw = U[3:6, idx_t0:].T
        X_raw = U[0:3, idx_t0:].T
        Z_raw = U[6:9, idx_t0:].T
        
        XZ_raw = torch.tensor(np.hstack([X_raw, Z_raw]), dtype=torch.float32)
        Y_raw = torch.tensor(Y_raw, dtype=torch.float32)

        # 2. Split the data in train and validation
        # Also compute normalization statistics (using only train)
        Nt = len(Y_raw)
        split_idx = int(Nt * train_ratio)

        y_train = Y_raw[:split_idx]
        xz_train = XZ_raw[:split_idx]

        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std = y_train.std(dim=0, keepdim=True)

        xz_mean = xz_train.mean(dim=0, keepdim=True)
        xz_std = xz_train.std(dim=0, keepdim=True)

        if save_stats and split == 'train':
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            stats = {
                'y_mean': y_mean, 'y_std': y_std,
                'xz_mean': xz_mean, 'xz_std': xz_std,
                'x_mean': xz_mean[:, :3], 'x_std': xz_std[:, :3],
                'z_mean': xz_mean[:, 3:], 'z_std': xz_std[:, 3:],
                'N_hist': N_hist,
                'stride': stride,
                'dt': dt,
                'k_points': N_hist // stride,
                'split_idx': split_idx + idx_t0
            }
            torch.save(stats, stats_path)

        # 3. Return normalized data
        if split == 'train':
            self.Y_norm = (Y_raw[:split_idx] - y_mean) / y_std
            self.XZ_norm = (XZ_raw[:split_idx] - xz_mean) / xz_std
        elif split == 'val' or split == 'test':
            # Incluimos los N_hist puntos anteriores al corte para no perder el inicio del test set
            self.Y_norm = (Y_raw[split_idx - N_hist:] - y_mean) / y_std
            self.XZ_norm = (XZ_raw[split_idx - N_hist:] - xz_mean) / xz_std

        self.N_hist = N_hist
        self.stride = stride

    def __len__(self):
        return len(self.Y_norm) - self.N_hist

    def __getitem__(self, idx):
        t = idx + self.N_hist
        y_hist = self.Y_norm[t - self.N_hist : t : self.stride]
        y_cond = y_hist.reshape(-1)
        target_xz = self.XZ_norm[t]
        return y_cond, target_xz


def get_dataloaders(file_path='data/NN_training_data.mat', batch_size=1024, 
                    N_hist=5000, stride=50, train_ratio=0.8):
    
    train_dataset = HistoryDataset(file_path=file_path, N_hist=N_hist, stride=stride, split='train', train_ratio=train_ratio)
    val_dataset = HistoryDataset(file_path=file_path, N_hist=N_hist, stride=stride, split='val', train_ratio=train_ratio, save_stats=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader