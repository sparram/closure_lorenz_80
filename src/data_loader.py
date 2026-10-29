import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from config import data_path, idx_0, N_train, N_test, batch_size, K, stride, dt

torch.manual_seed(37)

class MemoryDataset(Dataset):
    def __init__(self, split, K, stride, t_transient = 10.0, split_idx = 1344000, val_points=576000, history_len = 5000, save_stats=True):

        data = scipy.io.loadmat(data_path)
        U_raw = data["u"]

        idx_start = int(t_transient / dt)
        X_raw = U_raw[0:3][:, idx_start:].T
        Y_raw = U_raw[3:6][:, idx_start:].T
        Z_raw = U_raw[6:9][:, idx_start:].T

        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        Y_raw = torch.tensor(Y_raw, dtype=torch.float32)
        Z_raw = torch.tensor(Z_raw, dtype=torch.float32)

        X_train = X_raw[:split_idx]
        Y_train = Y_raw[:split_idx]
        Z_train = Z_raw[:split_idx]

        meanX = X_train.mean(dim=0, keepdim=True)
        meanY = Y_train.mean(dim=0, keepdim=True)
        meanZ = Z_train.mean(dim=0, keepdim=True)

        stdX = X_train.std(dim=0, keepdim=True)
        stdY = Y_train.std(dim=0, keepdim=True)
        stdZ = Z_train.std(dim=0, keepdim=True)
        
        if save_stats and split == "train":    
            norm_stats={
                "meanX" : meanX, "stdX" : stdX,
                "meanY" : meanY, "stdY" : stdY,
                "meanZ" : meanZ, "stdZ" : stdZ
            }
            torch.save(norm_stats, "norm_stats.pt")

        if split == "train":
            self.X_norm = (X_raw - meanX) / stdX
            self.Y_norm = (Y_raw - meanY) / stdY
            self.Z_norm = (Z_raw - meanZ) / stdZ
        elif split == "val":
            end_idx = min(split_idx + val_points, len(Y_raw))
            self.X_norm = (X_raw[split_idx - history_len : end_idx] - meanX) / stdX
            self.Y_norm = (Y_raw[split_idx - history_len : end_idx] - meanY) / stdY
            self.Z_norm = (Z_raw[split_idx - history_len : end_idx] - meanZ) / stdZ

        self.history_len = history_len
        self.stride = stride
            
    def __len__(self):
        return len(self.Y_norm) - self.history_len
        
    def __getitem__(self, idx):
        idx_adj = idx + self.history_len
        Y_hist = self.Y_norm[idx_adj - self.history_len : idx_adj : self.stride]
        X_in = Y_hist.reshape(-1)
        X_out = torch.cat([self.X_norm[idx_adj, :], self.Z_norm[idx_adj, :]])
        #print(X_in.shape, X_out.shape)
        return X_in, X_out
        
def get_dataloaders(batch_size=1024, history_len=5000, stride=50, split_idx=1344000, val_points=576000):
    train_dataset = MemoryDataset(split="train", K=K, stride=stride)
    val_dataset = MemoryDataset(split="val", K=K, stride=stride, save_stats=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Generando lote de prueba - Versión Unificada (cfm)...")
    _, train_loader = get_dataloaders(batch_size=1024)
    
    # Tomar el primer lote
    inputs, targets = next(iter(train_loader))
    
    print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    
    # Guardar en un archivo temporal
    torch.save({'inputs': inputs, 'targets': targets}, 'batch_v2.pt')
    print("¡Guardado en 'batch_v2.pt'!")