import scipy.io
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.physics import rk4_step_y

def run_level1_validation(file_path='data/NN_training_data.mat', n_steps=50000, dt=4.2e-3):
    raw_data = scipy.io.loadmat(file_path)['u']
    
    # Conversión directa a arreglos de JAX
    X_true = jnp.array(raw_data[0:3, :n_steps].T)
    Y_true = jnp.array(raw_data[3:6, :n_steps].T)
    Z_true = jnp.array(raw_data[6:9, :n_steps].T)

    def step_fn(y_curr, inputs):
        x_t, z_t = inputs
        y_next = rk4_step_y(y_curr, x_t, z_t, dt=dt)
        return y_next, y_next

    # Integración rápida en JAX
    _, Y_sim = jax.lax.scan(step_fn, Y_true[0], (X_true[:-1], Z_true[:-1]))
    Y_sim = jnp.concatenate([Y_true[0:1], Y_sim], axis=0)

    mse = jnp.mean((Y_sim - Y_true) ** 2)
    print(f"[Nivel 1 JAX] Integración completada. MSE vs Real: {mse:.8e}")

    t_axis = jnp.arange(n_steps) * dt
    plt.figure(figsize=(12, 4))
    plt.plot(t_axis, Y_true[:, 0], 'k-', label='Y1 Real', alpha=0.8)
    plt.plot(t_axis, Y_sim[:, 0], 'r--', label='Y1 Integrado (Forzamiento Real X, Z)', alpha=0.8)
    plt.xlabel('Tiempo físico (t)')
    plt.ylabel('y1')
    plt.title('Nivel 1: Test de Forzamiento Perfecto (JAX)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_level1_validation()