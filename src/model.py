import torch
import torch.nn as nn
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


class ConditionalVelocityField(nn.Module):
    """
    Red para v_theta(z_tau, tau | y_history).
    dim_cond por defecto: 300 (100 puntos históricos x 3 componentes de Y)
    """
    def __init__(self, dim_fast=6, dim_cond=300, time_emb_dim=16, hidden_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Dimension de entrada: 6 (z_tau) + 16 (tau) + dim_cond (historia de Y)
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

    def forward(self, z_tau, tau, y_cond):
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
            
        t_emb = self.time_mlp(tau.squeeze(-1))
        x_in = torch.cat([z_tau, t_emb, y_cond], dim=-1)
        return self.net(x_in)