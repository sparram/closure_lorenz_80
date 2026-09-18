import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """ Embedding de frecuencias para el tiempo sintetico tau """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalVelocityField(nn.Module):
    """
    Red que aproxima el campo de vectores v_theta(z_tau, tau | y).
    
    Entradas:
        z_tau: Tensor (batch_size, 6) -> Estado en el tiempo sintético
        tau:   Tensor (batch_size, 1) o (batch_size,) -> Tiempo sintético en [0, 1]
        y:     Tensor (batch_size, 3) -> Estado de la variable lenta
    """
    def __init__(self, dim_fast=6, dim_cond=3, time_emb_dim=16, hidden_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Dimension total de entrada: 6 (z_tau) + 16 (tau embedded) + 3 (y) = 25
        in_dim = dim_fast + time_emb_dim + dim_cond

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_fast)
        )

    def forward(self, z_tau, tau, y):
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
            
        t_emb = self.time_mlp(tau.squeeze(-1))
        
        # Concatenacion: [z_tau, embedding(tau), y]
        x_in = torch.cat([z_tau, t_emb, y], dim=-1)
        return self.net(x_in)


# Test unitario de dimensiones
if __name__ == "__main__":
    batch_size = 32
    z_tau = torch.randn(batch_size, 6)
    tau = torch.rand(batch_size, 1)
    y = torch.randn(batch_size, 3)

    model = ConditionalVelocityField()
    out = model(z_tau, tau, y)
    print("Shape de salida:", out.shape)  # Debe ser torch.Size([32, 6])