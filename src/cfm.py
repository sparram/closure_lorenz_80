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
        # input_samples.shape : (N_samples, 3*K)
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

    @torch.no_grad()
    def tail_sample(self, input_samples, N_noise=32, N_steps=cfm_steps, alpha=1.0, length_scale=0.5):
        """
        Diversity-Enhancing Rejection Sampling (Rejection + Greedy k-DPP)
        input_samples.shape : (N_ens, 3*K)
        """
        self.model.eval()
        N_ens = input_samples.shape[0]
        device = input_samples.device

        # 1. Generar pool de candidatos en paralelo: (N_ens * N_noise, 6)
        ext_samples = input_samples.repeat_interleave(N_noise, dim=0)
        z_tau = torch.randn(N_ens * N_noise, 6, device=device)
        d_tau = 1.0 / N_steps

        for step in range(N_steps):
            tau = torch.full((N_ens * N_noise, 1), step * d_tau, device=device)
            v_flow = self.model(z_tau, tau, ext_samples)
            z_tau += d_tau * v_flow

        states = z_tau  # (M, 6) donde M = N_ens * N_noise
        
        # 2. REJECTION SAMPLING (Filtro hacia la cola extrema)
        R_ind = torch.norm(states, dim=-1)
        R_norm = (R_ind - R_ind.mean()) / (R_ind.std() + 1e-8)
        
        # Probabilidad de aceptación suave hacia eventos extremos (evita colapso)
        # Mantiene también parte de la distribución base para no perder soporte
        log_prob = alpha * R_norm
        accept_prob = torch.exp(log_prob - log_prob.max())
        # Mezcla 70% cola / 30% uniforme para asegurar cobertura global + extremos
        accept_prob = 0.7 * accept_prob + 0.3 * torch.ones_like(accept_prob)
        
        u = torch.rand_like(accept_prob)
        accepted_mask = u < accept_prob
        
        # Salvaguarda: asegurar al menos N_ens candidatos aceptados
        if accepted_mask.sum() < N_ens:
            _, top_idx = torch.topk(R_norm, k=min(N_ens * 4, states.shape[0]))
            pool_states = states[top_idx]
            pool_R = R_norm[top_idx]
        else:
            pool_states = states[accepted_mask]
            pool_R = R_norm[accepted_mask]

        # 3. GREEDY k-DPP (Selección con repulsión para maximizar diversidad en el ensamble)
        # L_ij = q_i * K(s_i, s_j) * q_j
        M_pool = pool_states.shape[0]
        q = torch.exp(0.5 * alpha * pool_R)  # Calidad (energía) de cada candidato
        
        dist_sq = torch.cdist(pool_states, pool_states, p=2.0) ** 2
        K_sim = torch.exp(-dist_sq / (2.0 * length_scale ** 2))
        L = (q[:, None] * K_sim) * q[None, :]  # Matriz de kernel DPP (M_pool, M_pool)

        # Algoritmo Greedy k-DPP (Chen et al., NeurIPS - O(M_pool * N_ens^2) ultra rápido)
        selected_idxs = []
        diag_L = torch.diagonal(L).clone()
        cis = torch.zeros((N_ens, M_pool), device=device)

        for k in range(N_ens):
            # Elegir el candidato que maximiza el determinante marginal (calidad * diversidad)
            idx = torch.argmax(diag_L).item()
            selected_idxs.append(idx)
            
            if k == N_ens - 1:
                break
                
            # Actualizar proyección ortogonal para penalizar candidatos cercanos al recién elegido
            e_i = L[idx, :]
            if k > 0:
                proj = torch.mv(cis[:k, :].T, cis[:k, idx])
                e_i = e_i - proj
            
            norm_val = torch.sqrt(torch.clamp(diag_L[idx], min=1e-12))
            cis[k, :] = e_i / norm_val
            diag_L = diag_L - cis[k, :] ** 2
            diag_L[selected_idxs] = -float('inf')  # Evitar reemplazo (cero clonación)

        selected_idxs = torch.tensor(selected_idxs, device=device, dtype=torch.long)
        chosen = pool_states[selected_idxs]
        
        return chosen[:, 0:3], chosen[:, 3:6]

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