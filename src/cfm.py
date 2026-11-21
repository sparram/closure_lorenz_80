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
    def div_enhacing_sample(self, input_samples, N_noise=32, N_steps=cfm_steps, alpha=0.1, length_scale=1.25):
        # input_samples.shape : (N_samples, 3*K)
        self.model.eval()
        N_ens = input_samples.shape[0]
        device = input_samples.device

        ext_samples = input_samples.repeat_interleave(N_noise, dim=0)
        original_idxs = torch.arange(N_ens, device=device).repeat_interleave(N_noise)
        
        z_tau = torch.randn(N_ens * N_noise, 6, device=device)
        d_tau = 1.0 / N_steps

        for step in range(N_steps):
            tau = torch.full((N_ens * N_noise, 1), step * d_tau, device=device)
            v_flow = self.model(z_tau, tau, ext_samples)
            z_tau += d_tau * v_flow

        # sampled_xz.shape : (N_ens * N_noise, 6)
        sampled_xz = z_tau 

        # REJECTION SAMPLING: EVT THEORY
        R_ind = torch.norm(sampled_xz, dim=-1)
        R_norm = (R_ind - R_ind.mean()) / (R_ind.std() + 1e-8)

        log_prob = alpha * R_norm
        accept_prob = torch.exp( -(log_prob.max() - log_prob) )
        # Trick to not discard low probability (standard) events
        accept_prob = 0.8 * accept_prob + 0.2 * torch.ones_like(accept_prob)

        u = torch.rand_like(accept_prob) # Rejection sampling
        is_accepted = u < accept_prob

        # We define a pool of non-rejected samples
        # The pool will contain at least N_ens elements
        if is_accepted.sum() >= N_ens:
            pool_states = sampled_xz[is_accepted]
            pool_R = R_norm[is_accepted]
            pool_parents = original_idxs[is_accepted]
        else:
            # This may never happen, but we add a safety pool just in case
            min_pool = 4 * N_ens
            _, topk_idxs = torch.topk(R_norm, k=min_pool)
            pool_states = sampled_xz[topk_idxs]
            pool_R = R_norm[topk_idxs]
            pool_parents = original_idxs[topk_idxs]

        # GREEDY k-DPP
        M_pool = pool_states.shape[0] # Pool size
        weights = torch.exp(0.5 * alpha * pool_R)

        # dist_K shape: (M_pool, M_pool)
        dist_sq = torch.cdist(pool_states, pool_states, p=2.0) ** 2
        dist_K = torch.exp(-dist_sq / (2.0 * length_scale ** 2))
        
        # DPP Kernel shape: (M_pool, M_pool)
        L = (weights[:, None] * dist_K) * weights[None, :]

        # GREEDY SELECTION
        selected_idxs = []
        diag_L = torch.diagonal(L).clone()
        cis = torch.zeros((N_ens, M_pool), device=device)

        for k in range(N_ens):
            # Take the candidate which maximizes the actual L
            idx = torch.argmax(diag_L).item()
            selected_idxs.append(idx)

            # Stop if we finished
            if k == N_ens - 1:
                break
                
            # Update energy in the actual idx
            e_i = L[idx, :]
            if k > 0:
                proj = torch.mv(cis[:k, :].T, cis[:k, idx])
                e_i = e_i - proj

            # Update repulsion
            norm_val = torch.sqrt(torch.clamp(diag_L[idx], min=1e-12))
            cis[k, :] = e_i / norm_val
            diag_L -= cis[k, :] ** 2
            diag_L[selected_idxs] = -float('inf')

        selected_idxs = torch.tensor(selected_idxs, device=device, dtype=torch.long)
        chosen = pool_states[selected_idxs]
        chosen_parents = pool_parents[selected_idxs]

        return chosen[:, 0:3], chosen[:, 3:6], chosen_parents


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