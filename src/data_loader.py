import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from config import data_path, idx_0, N_train, batch_size, K, stride

torch.manual_seed(37)

class MemoryDataset(Dataset):
    def __init__(self, K, stride):
        X_in, X_out = load_data()
        self.X_in = X_in
        self.X_out = X_out
        self.N_data = N_train
        self.K = K
        self.stride = stride        
    
    def __len__(self):
        return self.N_data - self.K * self.stride
        
    def __getitem__(self, idx):
        idx += self.K * self.stride
        #idx = min(N_train-1, idx)
        Y_hist = self.X_in[idx - self.K * self.stride : idx : self.stride]
        X_in = Y_hist.reshape(-1)
        X_out = self.X_out[idx, :]
        #print(X_in.shape, X_out.shape)
        return X_in, X_out
        
def load_data():
    data = scipy.io.loadmat(data_path)
    U_raw = data["u"]
    
    #idx_0 = 100000
    #N_train = 1000000
    X_raw = U_raw[0:3][:, idx_0 : idx_0 + N_train]
    Y_raw = U_raw[3:6][:, idx_0 : idx_0 + N_train]
    Z_raw = U_raw[6:9][:, idx_0 : idx_0 + N_train]
    
    X = torch.from_numpy(X_raw)
    Y = torch.from_numpy(Y_raw)
    Z = torch.from_numpy(Z_raw)
    
    meanX, stdX = X.mean(dim=-1), X.std(dim=-1)
    meanY, stdY = Y.mean(dim=-1), Y.std(dim=-1)
    meanZ, stdZ = Z.mean(dim=-1), Z.std(dim=-1)

    stats = [meanX, stdX, meanY, stdY, meanZ, stdZ]
    save_norm_stats(stats)
    
    X = (X - meanX[:, None]) / stdX[:, None]
    Y = (Y - meanY[:, None]) / stdY[:, None]
    Z = (Z - meanZ[:, None]) / stdZ[:, None]
    
    X_in = Y.t()
    X_out = torch.cat([X, Z], dim=0).t()
    print(f"Datashape - In : {X_in.shape}, Out: {X_out.shape}")
            
    return X_in, X_out

def save_norm_stats(stats):
    meanX, stdX = stats[0:2]
    meanY, stdY = stats[2:4]
    meanZ, stdZ = stats[4:6]
    norm_stats={
        "meanX" : meanX, "stdX" : stdX,
        "meanY" : meanY, "stdY" : stdY,
        "meanZ" : meanZ, "stdZ" : stdZ
    }
    torch.save(norm_stats, "norm_stats.pt")

def get_dataloader():
    X_in, X_out = load_data()
    dataset = MemoryDataset(K=K, stride=stride)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == "__main__":
    train_loader = get_dataloader()
    device = 'cpu'
    for input_batch, out_batch in train_loader:
        input_batch = input_batch.to(device, dtype=torch.float32)
        out_batch = out_batch.to(device, dtype=torch.float32)
        print("BATCH:", input_batch.shape, out_batch.shape)
        break
        
    print("CHECK!")