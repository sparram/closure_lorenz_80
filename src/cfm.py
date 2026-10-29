import torch
import torch.nn as nn
import math
from config import K, stride, cfm_steps, hidden_dim, N_ens, device

torch.manual_seed(37)

# CONDITIONAL FLOW MATCHING
class CFM:
    def __init__(self, model):
        self.model = model

    def compute_loss(self, input_batch, out_batch):
        N_batch = input_batch.shape[0]
        device = input_batch.device
        
        z0 = torch.randn_like(out_batch, device=device)
        z1 = out_batch
        tau = torch.rand((N_batch, 1), device=device)
        z_tau = (1.0 - tau) * z0 + tau * z1
        
        v_target = z1 - z0
        v_pred = self.model(z_tau, tau, input_batch)
        loss = nn.functional.mse_loss(v_pred, v_target)
        
        return loss

    @torch.no_grad()
    def sample(self, input_samples, N_steps=cfm_steps):
        self.model.eval()
        N_ens = input_samples.shape[0]
        device = input_samples.device
        
        z_tau = torch.randn(N_ens, 6, device=device)
        d_tau = 1.0 / N_steps

        for step in range(N_steps):
            tau = torch.full((N_ens, 1), step * d_tau, device=device)
            v_flow = self.model(z_tau, tau, input_samples)
            z_tau += d_tau * v_flow

        x_hat, z_hat = z_tau[:, 0:3], z_tau[:, 3:6]
        return x_hat, z_hat

# POSITIONAL EMBEDDING
class TauPosEmb(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.dim = time_emb_dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(- torch.arange(half_dim, device=device) * torch.tensor(10000).log() / (half_dim-1))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# VELOCITY FLOW MODEL
class FlowModel(nn.Module):
    def __init__(self, cond_dim=3*K, out_dim=6, time_emb_dim=16, hidden_dim=hidden_dim):
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

    def forward(self, z_tau, tau, cond):
        t_emb = self.time_mlp(tau.squeeze(-1))
        x_input = torch.cat([z_tau, t_emb, cond], dim=-1)
        return self.net(x_input)

if __name__ == "__main__":
    input_batch = torch.randn(N_ens, 300)
    out_batch = torch.randn(N_ens, 6)
    
    v_flow = FlowModel()
    cfm = CFM(v_flow)
    loss = cfm.compute_loss(input_batch, out_batch)
    print(loss)