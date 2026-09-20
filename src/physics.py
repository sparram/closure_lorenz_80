import math
import torch
import jax.numpy as jnp

# Parámetros físicos del sistema L80
a1, a2, a3 = 1.0, 1.0, 3.0
v0 = 1.0 / 48.0
b1 = (a1 - a2 - a3) / 2.0
b2 = (a2 - a3 - a1) / 2.0
b3 = (a3 - a1 - a2) / 2.0
c = math.sqrt(b1 * b2 + b2 * b3 + b3 * b1)

def dYdt(x, y, z):
    """ Ecuación diferencial dy/dt polimórfica para PyTorch y JAX"""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
    z1, z2, z3 = z[..., 0], z[..., 1], z[..., 2]
    dy1dt = (- a3 * b3 * x2 * y3 - a2 * b2 * y2 * x3 + c * (a3 - a2) * y2 * y3 - a1 * x1 - v0 * a1**2 * y1) / a1
    dy2dt = (- a1 * b1 * x3 * y1 - a3 * b3 * y3 * x1 + c * (a1 - a3) * y3 * y1 - a2 * x2 - v0 * a2**2 * y2) / a2
    dy3dt = (- a2 * b2 * x1 * y2 - a1 * b1 * y1 * x2 + c * (a2 - a1) * y1 * y2 - a3 * x3 - v0 * a3**2 * y3) / a3

    if isinstance(y, torch.Tensor):
        return torch.stack([dy1dt, dy2dt, dy3dt], dim=-1)
    
    else:
        return jnp.stack([dy1dt, dy2dt, dy3dt], axis=-1)

def rk4_step_y(y, x, z, dt=4.2e-3):
    """ Integrador RK4 para dy/dt """
    k1 = dYdt(x, y, z)
    k2 = dYdt(x, y + 0.5 * dt * k1, z)
    k3 = dYdt(x, y + 0.5 * dt * k2, z)
    k4 = dYdt(x, y + dt * k3, z)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)