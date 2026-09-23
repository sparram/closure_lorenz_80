import torch
import matplotlib.pyplot as plt

def plot_attractor_2d_projections():
    print("Cargando el historial de simulación...")
    data = torch.load('checkpoints/simulation_history_100k.pt', map_location='cpu')
    
    history_y = data['history_y'].numpy()  # Forma: [n_steps, n_ensemble, 3]
    Y_true = data['Y_true'].numpy()        # Forma: [n_steps, 3]
    
    # Media del ensamble para las variables Y
    mean_y = history_y.mean(axis=1)        # Forma: [n_steps, 3]
    
    print("Generando proyecciones 2D del atractor...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Pares de variables para los planos de fase
    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]
    
    for ax, (i, j), (lbl_i, lbl_j) in zip(axes, pairs, labels):
        # Graficar el atractor real en el plano 2D
        ax.plot(Y_true[:, i], Y_true[:, j], 
                color='black', alpha=0.5, linewidth=0.6, label='Atractor Real')
        
        # Graficar la simulación en bucle cerrado
        ax.plot(mean_y[:, i], mean_y[:, j], 
                color='red', alpha=0.7, linewidth=0.6, label='Bucle Cerrado')
        
        ax.set_xlabel(lbl_i)
        ax.set_ylabel(lbl_j)
        ax.set_title(f'Plano de Fase: {lbl_i} vs {lbl_j}')
        ax.legend(loc='upper right')
        ax.grid(True)
        
    plt.suptitle('Comparación del Atractor en Proyecciones 2D', fontsize=14)
    plt.tight_layout()
    plt.savefig('media/closed_loop_attractor_2d.png', dpi=300)
    print("¡Gráficas guardadas en media/closed_loop_attractor_2d.png!")
    plt.show()

if __name__ == '__main__':
    plot_attractor_2d_projections()