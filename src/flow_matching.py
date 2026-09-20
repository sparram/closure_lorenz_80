import torch
import torch.nn as nn
from src.physics import dYdt  # Importamos la física que ya tienes

class ConditionalFlowMatcher:
    def __init__(self, model, lambda_physics=0.1):
        self.model = model
        self.lambda_physics = lambda_physics  # Ponderación del término físico

    def compute_loss(self, y_batch, fast_batch):
        """
        Calcula la pérdida híbrida: MSE de Flow Matching + Consistencia Física (PINN).
        
        y_batch: Tensor (batch_size, 3) -> Estado lento real
        fast_batch: Tensor (batch_size, 6) -> Variables rápidas reales [x, z] (z1)
        """
        batch_size = y_batch.shape[0]
        device = y_batch.device

        # --- 1. PÉRDIDA CFM ESTÁNDAR (Transporte Óptimo) ---
        z0 = torch.randn_like(fast_batch)
        z1 = fast_batch
        tau = torch.rand(batch_size, 1, device=device)

        z_tau = (1.0 - tau) * z0 + tau * z1
        target_velocity = z1 - z0  

        pred_velocity = self.model(z_tau, tau, y_batch)
        loss_cfm = nn.functional.mse_loss(pred_velocity, target_velocity)

        # --- 2. PÉRDIDA FÍSICA (Estilo PINN usando physics.py) ---
        # Evaluamos la predicción al final del flujo sintético (tau = 1)
        z_pred_final = z_tau + pred_velocity * (1.0 - tau)
        
        hat_x = z_pred_final[:, 0:3]
        hat_z = z_pred_final[:, 3:6]

        # Separamos las variables reales para sacar la derivada física "verdadera"
        x_real = fast_batch[:, 0:3]
        z_real = fast_batch[:, 3:6]

        # Derivada de Y calculada con los datos reales
        dy_true = dYdt(x_real, y_batch, z_real)

        # Derivada de Y calculada con las variables rápidas predichas por la red
        dy_pred = dYdt(hat_x, y_batch, hat_z)

        # Penalizamos la discrepancia física
        loss_physics = nn.functional.mse_loss(dy_pred, dy_true)

        # --- 3. PÉRDIDA TOTAL COMBINADA ---
        total_loss = loss_cfm + self.lambda_physics * loss_physics

        return total_loss

    @torch.no_grad()
    def sample(self, y_ensemble, steps=5):
        # El método de muestreo se mantiene idéntico
        self.model.eval()
        num_samples = y_ensemble.shape[0]
        device = y_ensemble.device

        z_tau = torch.randn(num_samples, 6, device=device)
        d_tau = 1.0 / steps

        for step in range(steps):
            tau_val = torch.full((num_samples, 1), step * d_tau, device=device)
            v_pred = self.model(z_tau, tau_val, y_ensemble)
            z_tau = z_tau + v_pred * d_tau

        hat_x = z_tau[:, 0:3]
        hat_z = z_tau[:, 3:6]
        return hat_x, hat_z