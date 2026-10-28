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

        z0 = torch.randn_like(out_batch)
        z1 = out_batch
        tau = torch.rand(N_batch, 1, device=device)
        z_tau = (1.0 - tau) * z0 + tau * z1
        
        v_pred = z1 - z0
        v_target = self.model(z_tau, tau, input_batch)
        loss = nn.functional.mse_loss(v_pred, v_target)
        
        return loss

    @torch.no_grad()
    def sample(self, y_cond_ens, steps=cfm_steps):
        self.model.eval()
        N_ens = y_cond_ens.shape[0]
        device = y_cond_ens.device

        z_tau = torch.randn(N_ens, 6, device=device)
        d_tau = 1.0 / steps

        for step in range(steps):
            tau = torch.full((N_ens, 1), step * d_tau, device=device)
            v_flow = self.model(z_tau, tau, y_cond_ens)
            z_tau += d_tau * v_flow

        x_hat, z_hat = z_tau[:, 0:3], z_tau[:, 3:6]
        return x_hat, z_hat