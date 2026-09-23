import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def plot_hlf_attractor_3d():
    print("Cargando datos del régimen HLF...")
    # Asegúrate de que la ruta al archivo .mat sea la correcta para tu estructura de carpetas
    data = scipy.io.loadmat('data/NN_training_data.mat')
    X = data['u']  # Forma esperada: (9, n_steps)
    
    # Selecciona el rango de tiempo que deseas visualizar (por ejemplo, omitiendo transitorios si es necesario)
    # Aquí tomamos una buena cantidad de puntos para ver la estructura del atractor
    start_idx = 4000000
    end_idx = 5000000 # X.shape[1]  # O puedes poner un número fijo, ej. 50000 para que cargue más rápido
    
    # Define qué variables quieres cruzar en el espacio 3D 
    # (En el modelo L80, y1, y2, y3 suelen estar en los índices 3, 4, 5 o x en 0,1,2)
    idx1, idx2, idx3 = 3, 4, 5  
    
    print(f"Generando gráfico 3D usando las variables {idx1}, {idx2} y {idx3}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar la trayectoria en 3D
    ax.plot(
        X[idx1, start_idx:end_idx], 
        X[idx2, start_idx:end_idx], 
        X[idx3, start_idx:end_idx], 
        color='blue', alpha=0.5, linewidth=0.4
    )
    
    ax.set_title('Atractor 3D - Régimen HLF (L80)', fontsize=14)
    ax.set_xlabel(f'Variable {idx1}')
    ax.set_ylabel(f'Variable {idx2}')
    ax.set_zlabel(f'Variable {idx3}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_hlf_attractor_3d()