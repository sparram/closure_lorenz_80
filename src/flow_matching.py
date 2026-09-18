import torch
import torch.nn as nn

class ConditionalFlowMatcher:
    """
    Maneja el entrenamiento con Optimal Transport Flow Matching (OT-FM)
    y la integración de la ODE en tiempo sintético tau para inferencia.
    """
    def __init__(self, model):
        self.model = model

    def compute_loss(self, y_batch, fast_batch):
        """
        Calcula la pérdida MSE sobre el campo de velocidades v_theta.
        
        y_batch: Tensor (batch_size, 3) -> Estado lento
        fast_batch: Tensor (batch_size, 6) -> Variables rápidas reales [x, z] (z1)
        """
        batch_size = y_batch.shape[0]
        device = y_batch.device

        # 1. Muestrear ruido gaussiano z0 ~ N(0, I) y el punto final real z1
        z0 = torch.randn_like(fast_batch)
        z1 = fast_batch

        # 2. Muestrear tiempo sintético tau ~ Uniforme(0, 1)
        # ESTRATEGIA DEL NN1
        tau = torch.rand(batch_size, 1, device=device)

        # 3. Trayectoria de Transporte Óptimo (Línea recta entre ruido y datos)
        z_tau = (1.0 - tau) * z0 + tau * z1
        target_velocity = z1 - z0  # Derivada dz_tau / d_tau

        # 4. Predicción del modelo y pérdida MSE
        pred_velocity = self.model(z_tau, tau, y_batch)
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)

        return loss

    @torch.no_grad()
    def sample(self, y_ensemble, steps=5):
        """
        Resuelve la ODE sintética d(z)/d(tau) = v_theta desde tau=0 hasta tau=1.
        
        y_ensemble: Tensor (N_ensemble, 3) -> Estado actual del ensemble
        steps: Pasos de integración numéricos en el tiempo sintético tau
        
        Retorna:
            hat_x: Tensor (N_ensemble, 3)
            hat_z: Tensor (N_ensemble, 3)
        """
        self.model.eval()
        num_samples = y_ensemble.shape[0]
        device = y_ensemble.device

        # Estado inicial en tau = 0: Ruido puro N(0, I)
        z_tau = torch.randn(num_samples, 6, device=device)
        d_tau = 1.0 / steps

        # Integrador Euler en tiempo sintético tau
        for step in range(steps):
            tau_val = torch.full((num_samples, 1), step * d_tau, device=device)
            v_pred = self.model(z_tau, tau_val, y_ensemble)
            z_tau = z_tau + v_pred * d_tau

        # Separar en componentes físicas (3 para x, 3 para z)
        hat_x = z_tau[:, 0:3]
        hat_z = z_tau[:, 3:6]
        return hat_x, hat_z