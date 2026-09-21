import scipy.io
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.physics import rk4_step_y

def run_level1_validation(file_path='data/NHLR_data.mat', n_steps=50000, dt=4.2e-3):
    raw_data = scipy.io.loadmat(file_path)['u']
    
    # Direct conversion to JAX arrays
    X_true = jnp.array(raw_data[0:3, :n_steps].T)
    Y_true = jnp.array(raw_data[3:6, :n_steps].T)
    Z_true = jnp.array(raw_data[6:9, :n_steps].T)

    def step_fn(y_curr, inputs):
        x_t, z_t = inputs
        y_next = rk4_step_y(y_curr, x_t, z_t, dt=dt)
        return y_next, y_next

    # Fast integration using JAX scan
    _, Y_sim = jax.lax.scan(step_fn, Y_true[0], (X_true[:-1], Z_true[:-1]))
    Y_sim = jnp.concatenate([Y_true[0:1], Y_sim], axis=0)

    mse = jnp.mean((Y_sim - Y_true) ** 2)
    print(f"[Level 1 JAX] Integration completed. MSE vs Real: {mse:.8e}")

    # --- MULTIPANEL PLOTTING (Y1, Y2, Y3) ---
    t_axis = jnp.arange(n_steps) * dt * 0.000520833
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Component Y1
    axes[0].plot(t_axis, Y_true[:, 0], 'k-', label='Real Y1', alpha=0.8)
    axes[0].plot(t_axis, Y_sim[:, 0], 'r--', label='Integrated Y1 (with Real X, Z)', alpha=0.8)
    axes[0].set_ylabel('Y1')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Component Y2
    axes[1].plot(t_axis, Y_true[:, 1], 'k-', label='Real Y2', alpha=0.8)
    axes[1].plot(t_axis, Y_sim[:, 1], 'r--', label='Integrated Y1 (with Real X, Z)', alpha=0.8)
    axes[1].set_ylabel('Y2')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # Component Y3
    axes[2].plot(t_axis, Y_true[:, 2], 'k-', label='Real Y3', alpha=0.8)
    axes[2].plot(t_axis, Y_sim[:, 2], 'r--', label='Integrated Y1 (with Real X, Z)', alpha=0.8)
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Y3')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.suptitle('Physics Test (JAX) - Slow Variables Y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_level1_validation()