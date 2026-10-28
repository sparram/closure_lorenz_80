import torch
import torch.nn as nn

class ConditionalFlowMatcher:
    """
    Maneja el entrenamiento con Optimal Transport Flow Matching (OT-FM)
    y la integración de la ODE en tiempo sintético tau para inferencia.
    """
    def __init__(self, model):
        self.model = model

    def compute_loss(self, y_cond_batch, fast_batch):
        """
        Calcula la pérdida MSE sobre el campo de velocidades v_theta.
        
        y_cond_batch: Tensor (batch_size, 300) -> Historial aplanado de Y
        fast_batch: Tensor (batch_size, 6) -> Variables rápidas reales [x, z]
        """
        batch_size = y_cond_batch.shape[0]
        device = y_cond_batch.device

        z0 = torch.randn_like(fast_batch)
        z1 = fast_batch

        tau = torch.rand(batch_size, 1, device=device)

        z_tau = (1.0 - tau) * z0 + tau * z1
        target_velocity = z1 - z0

        pred_velocity = self.model(z_tau, tau, y_cond_batch)
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)

        return loss

    @torch.no_grad()
    def sample(self, y_cond_ensemble, steps=5):
        """
        Resuelve la ODE sintética d(z)/d(tau) = v_theta desde tau=0 hasta tau=1.
        
        y_cond_ensemble: Tensor (N_ensemble, 300) -> Historial aplanado del ensemble
        steps: Pasos de integración numéricos en tiempo sintético tau
        """
        self.model.eval()
        num_samples = y_cond_ensemble.shape[0]
        device = y_cond_ensemble.device

        z_tau = torch.randn(num_samples, 6, device=device)
        d_tau = 1.0 / steps

        for step in range(steps):
            tau_val = torch.full((num_samples, 1), step * d_tau, device=device)
            v_pred = self.model(z_tau, tau_val, y_cond_ensemble)
            z_tau = z_tau + v_pred * d_tau

        hat_x = z_tau[:, 0:3]
        hat_z = z_tau[:, 3:6]
        return hat_x, hat_z