import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import data_path, stats_path, T0, idx_0, dt, N_hist, N_train, N_test, batch_size, K, stride

class MemoryDataset(Dataset):
    def __init__(self, split='train'):

        data = scipy.io.loadmat(data_path)
        U = data['u']  

        idx_t0 = int(T0 / dt)
        X_raw = U[0:3, idx_t0:].T
        Y_raw = U[3:6, idx_t0:].T
        Z_raw = U[6:9, idx_t0:].T

        XZ = torch.tensor(np.hstack([X_raw, Z_raw]), dtype=torch.float32)
        Y = torch.tensor(Y_raw, dtype=torch.float32)
        
        Nt = len(Y)
        split_idx = N_train

        XZtrain = XZ[:N_train]
        Ytrain = Y[:N_train]
        
        meanXZ, stdXZ = XZtrain.mean(dim=0, keepdim=True), XZtrain.std(dim=0, keepdim=True)
        meanY, stdY = Ytrain.mean(dim=0, keepdim=True), Ytrain.std(dim=0, keepdim=True)

        if split == 'train':
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            stats ={
                "x_mean" : meanXZ[:, :3], "x_std" : stdXZ[:, :3],
                "y_mean" : meanY, "y_std" : stdY,
                "z_mean" : meanXZ[:, 3:], "z_std" : stdXZ[:, 3:],
                "xz_mean" : meanXZ, "xz_std" : stdXZ,
                'N_hist': N_hist, 'stride': stride,
                'dt': dt, 'k_points': N_hist // stride,
                'split_idx': split_idx + idx_t0
            }
            torch.save(stats, stats_path)

        # 3. Return normalized data
        if split == 'train':
            self.Y_norm = (Y[:N_train] - meanY) / stdY
            self.XZ_norm = (XZ[:N_train] - meanXZ) / stdXZ
        elif split == 'val' or split == 'test':
            # Incluimos los N_hist puntos anteriores al corte para no perder el inicio del test set
            self.Y_norm = (Y[N_train - N_hist:] - meanY) / stdY
            self.XZ_norm = (XZ[N_train - N_hist:] - meanXZ) / stdXZ

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

def get_dataloaders(batch_size=1024, 
                    N_hist=5000, stride=50):
    
    train_dataset = MemoryDataset(split='train')
    val_dataset = MemoryDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader