import torch
import torch.nn as nn
import math
from config import K, stride, cfm_steps, hidden_dim, N_ens, device

# POSITIONAL EMBEDDING
class TauPosEmb(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.dim = time_emb_dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(- torch.arange(half_dim, device=device) * math.log(10000) / (half_dim - 1))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# VELOCITY FLOW MODEL
class FlowModel(nn.Module):
    def __init__(self, out_dim=6, cond_dim=3*K, time_emb_dim=16, hidden_dim=hidden_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            TauPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        input_dim = out_dim + cond_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z_tau, tau, y_cond):
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        t_emb = self.time_mlp(tau.squeeze(-1))
        x_input = torch.cat([z_tau, t_emb, y_cond], dim=-1)
        return self.net(x_input)