import math
import torch
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# 1. Cargar pesos exportados del modelo PyTorch
data_weights = np.load('checkpoints/cfm_weights.npz')

def jax_sinusoidal_embedding(tau, dim=16):
    """Réplica exacta en JAX de SinusoidalPosEmb"""
    half_dim = dim // 2
    emb_scale = math.log(10000.0) / (half_dim - 1)
    # Corrección de la multiplicación de rangos
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb_scale)
    
    # Asegurar que tau sea al menos 1D para evitar errores con [:, None]
    tau = jnp.atleast_1d(tau)
    
    emb = tau[:, None] * emb[None, :]
    return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

def jax_mlp_forward(params, z_tau, tau, y):
    """
    Réplica exacta de ConditionalVelocityField en JAX, incluyendo time_mlp.
    """
    # Forzar tau a un formato 1D seguro
    tau_1d = jnp.atleast_1d(tau)
        
    # 1. Simular el bloque time_mlp completo de PyTorch
    t_emb_raw = jax_sinusoidal_embedding(tau_1d, dim=16)
    t_emb = jax.nn.silu(jnp.dot(t_emb_raw, params['time_mlp_1_weight'].T) + params['time_mlp_1_bias'])
    
    # 2. Asegurar formas limpias 1D para concatenar
    z_tau = jnp.squeeze(z_tau)  # (6,)
    t_emb = jnp.squeeze(t_emb)  # (16,)
    y = jnp.squeeze(y)          # (3,)
    
    # 3. Concatenación total: 6 + 16 + 3 = 25
    x_in = jnp.concatenate([z_tau, t_emb, y], axis=-1)
    
    # 4. Capas densas principales de la red (net)
    h1 = jax.nn.silu(jnp.dot(x_in, params['net_0_weight'].T) + params['net_0_bias'])
    h2 = jax.nn.silu(jnp.dot(h1, params['net_2_weight'].T) + params['net_2_bias'])
    h3 = jax.nn.silu(jnp.dot(h2, params['net_4_weight'].T) + params['net_4_bias'])
    out = jnp.dot(h3, params['net_6_weight'].T) + params['net_6_bias']  # (6,)
    
    return jnp.squeeze(out[0:3]), jnp.squeeze(out[3:6])

def jax_cfm_sample(params, y_norm, x_mean, x_std, z_mean, z_std, rng_key, steps=5):
    """
    Réplica exacta del método .sample(steps=...) de PyTorch en JAX puro.
    """
    # 1. Ruido inicial z_0 (equivalente a torch.randn)
    z_tau = jax.random.normal(rng_key, (6,))
    d_tau = 1.0 / steps

    # 2. Función de integración para el flujo sintético tau de 0 a 1
    def flow_step(z_current, step_idx):
        tau_val = step_idx * d_tau
        
        # Inferencia de la red con el tiempo sintético actual
        hat_x_norm, hat_z_norm = jax_mlp_forward(params, z_current, tau_val, y_norm)
        v_pred = jnp.concatenate([hat_x_norm, hat_z_norm], axis=-1)
        
        # Actualización de Euler para el flujo
        z_next = z_current + v_pred * d_tau
        return z_next, None

    # Ejecutar los 'steps' del flujo con lax.scan
    z_final, _ = jax.lax.scan(flow_step, z_tau, jnp.arange(steps, dtype=jnp.float32) * d_tau)
    
    # Extraer X y Z normalizados
    hat_x_norm = z_final[0:3]
    hat_z_norm = z_final[3:6]
    
    # Desnormalizar
    hat_x = hat_x_norm * x_std + x_mean
    hat_z = hat_z_norm * z_std + z_mean
    
    return hat_x, hat_z
    
@jax.jit
def jax_rk4_step_y(y, hat_x, hat_z, dt):
    """
    Paso RK4 para las variables lentas Y acopladas con las predichas X y Z.
    """
    def dYdt(state, x_v, z_v):
        y1, y2, y3 = state[..., 0], state[..., 1], state[..., 2]
        
        a1, a2, a3 = 1.0, 1.0, 3.0
        v0 = 1.0 / 48.0
        b1 = (a1 - a2 - a3) / 2.0
        b2 = (a2 - a3 - a1) / 2.0
        b3 = (a3 - a1 - a2) / 2.0
        c = math.sqrt(b1 * b2 + b2 * b3 + b3 * b1)

        dy1dt = (-a3 * b3 * x_v[..., 1] * y3 - a2 * b2 * y2 * x_v[..., 2] + c * (a3 - a2) * y2 * y3 - a1 * x_v[..., 0] - v0 * a1**2 * y1) / a1
        dy2dt = (-a1 * b1 * x_v[..., 2] * y1 - a3 * b3 * y3 * x_v[..., 0] + c * (a1 - a3) * y3 * y1 - a2 * x_v[..., 1] - v0 * a2**2 * y2) / a2
        dy3dt = (-a2 * b2 * x_v[..., 0] * y1 - a1 * b1 * y1 * x_v[..., 1] + c * (a2 - a1) * y1 * y2 - a3 * x_v[..., 2] - v0 * a3**2 * y3) / a3

        return jnp.stack([dy1dt, dy2dt, dy3dt], axis=-1)

    k1 = dYdt(y, hat_x, hat_z)
    k2 = dYdt(y + 0.5 * dt * k1, hat_x, hat_z)
    k3 = dYdt(y + 0.5 * dt * k2, hat_x, hat_z)
    k4 = dYdt(y + dt * k3, hat_x, hat_z)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def run_fast_simulation():
    dt = 4.2e-3
    n_steps = 50000  # Puedes subirlo a 50000 o más cuando compruebes que todo va fluido
    skip_transient = 4000000
    
    # 1. Cargar datos reales de referencia para comparar en los gráficos
    raw_data = scipy.io.loadmat('data/NHLR_data.mat')['u']
    X_true = raw_data[0:3, skip_transient:skip_transient + n_steps].T
    Y_true = raw_data[3:6, skip_transient:skip_transient + n_steps].T
    Z_true = raw_data[6:9, skip_transient:skip_transient + n_steps].T

    # 2. Cargar estadísticas de normalización
    stats = torch.load('checkpoints/norm_stats.pt', weights_only=True)
    y_mean = jnp.array(stats['y_mean'].numpy())
    y_std = jnp.array(stats['y_std'].numpy())
    x_mean = jnp.array(stats['x_mean'].numpy())
    x_std = jnp.array(stats['x_std'].numpy())
    z_mean = jnp.array(stats['z_mean'].numpy())
    z_std = jnp.array(stats['z_std'].numpy())

    # Estado inicial basado en el primer punto de los datos reales
    y0 = jnp.array(Y_true[0])

    # Función de escaneo para jax.lax.scan
    def scan_step(y_current, key_input):
        y_norm = (y_current - y_mean) / y_std
        
        key, subkey = jax.random.split(key_input)
        
        # Llamada al muestreo multietapa (ej. 5 pasos de flujo)
        hat_x, hat_z = jax_cfm_sample(
            data_weights, y_norm, x_mean, x_std, z_mean, z_std, subkey, steps=20
        )
        
        # Paso RK4 físico para Y
        next_y = jax_rk4_step_y(y_current, hat_x, hat_z, dt)
        next_y = jnp.reshape(next_y, (3,))  # Blindaje de forma
        
        return next_y, (next_y, hat_x, hat_z)

    print("Compilando y ejecutando simulación masiva ultrarrápida en JAX...")
    rng_key = jax.random.PRNGKey(42)
    _, (traj_y, traj_x, traj_z) = jax.lax.scan(scan_step, y0, jax.random.split(rng_key, n_steps))
    print("¡Simulación completada en tiempo récord!")

    # --- 3. PROCESAR Y GRAFICAR RESULTADOS ---
    print("Generando gráficos de validación...")
    t_axis = np.arange(n_steps) * dt
    
    # Convertir tensores de JAX a Numpy para matplotlib
    mean_y = np.array(traj_y)
    mean_x = np.array(traj_x)
    mean_z = np.array(traj_z)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Componente Y1
    axes[0].plot(t_axis, mean_y[:, 0], 'r--', label='JAX Pred Mean Y1')
    axes[0].plot(t_axis, Y_true[:, 0], 'k-', label='Real Y1', alpha=0.8)
    #axes[0].set_ylim(-0.2, 0.35)
    axes[0].set_ylabel('Y1 (Lento)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Componente X1
    axes[1].plot(t_axis, mean_x[:, 0], 'b--', label='CFM Pred X1')
    axes[1].plot(t_axis, X_true[:, 0], 'k-', label='Real X1', alpha=0.8)
    #axes[1].set_ylim(-0.012, 0.0)
    axes[1].set_ylabel('X1 (Rápido)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # Componente Z1
    axes[2].plot(t_axis, mean_z[:, 0], 'g--', label='CFM Pred Z1')
    axes[2].plot(t_axis, Z_true[:, 0], 'k-', label='Real Z1', alpha=0.8)
    #axes[2].set_ylim(-0.1, 0.3)
    axes[2].set_xlabel('Tiempo físico (t)')
    axes[2].set_ylabel('Z1 (Rápido)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.suptitle('Simulación en Lazo Cerrado con JAX (Nivel 3)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_fast_simulation()