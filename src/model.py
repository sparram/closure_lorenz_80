import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOConditionalVelocityField(nn.Module):
    """
    Campo de velocidades condicionado con FNO 1D preservando altas frecuencias
    (sin Global Average Pooling destructivo).
    """
    def __init__(self, dim_fast=6, in_channels=3, modes=16, width=64, time_emb_dim=16, k_points=100):
        super().__init__()
        self.modes = modes
        self.width = width
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)

        # En lugar de promediar, aplanamos la salida del FNO (width * k_points)
        fno_out_dim = self.width * k_points

        self.fc1 = nn.Linear(fno_out_dim + dim_fast + time_emb_dim, 256)
        self.fc2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, dim_fast)
        )

    def forward(self, z_tau, tau, y_hist):
        """
        z_tau: Tensor (batch_size, 6)
        tau: Tensor (batch_size, 1)
        y_hist: Tensor (batch_size, 3, k_points)
        """
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
            
        t_emb = self.time_mlp(tau.squeeze(-1)) # (batch, time_emb_dim)

        # Procesamiento FNO
        x = y_hist.permute(0, 2, 1) # (batch, k_points, 3)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)      # (batch, width, k_points)

        x1 = F.gelu(self.conv0(x) + self.w0(x))
        x2 = F.gelu(self.conv1(x1) + self.w1(x1)) # (batch, width, k_points)
        
        # Aplanamos conservando toda la información espacial/temporal espectral
        x_fno = x2.reshape(x2.shape[0], -1) # (batch, width * k_points)

        # Concatenar con estado rápido y tiempo
        combined = torch.cat([x_fno, z_tau, t_emb], dim=-1)

        out = self.fc1(combined)
        return self.fc2(out)