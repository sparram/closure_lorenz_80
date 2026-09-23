import torch
import matplotlib.pyplot as plt

def plot_attractor_2d_projections():
    print("Loading simulation history...")
    data = torch.load('checkpoints/simulation_history_100k.pt', map_location='cpu')
    
    history_y = data['history_y'].numpy()  # Shape: [n_steps, n_ensemble, 3]
    Y_true = data['Y_true'].numpy()        # Shape: [n_steps, 3]
    
    # Ensemble mean for Y variables
    mean_y = history_y.mean(axis=1)        # Shape: [n_steps, 3]
    
    print("Generating 2D projections of the attractor...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Variable pairs for phase planes
    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [('Y1', 'Y2'), ('Y2', 'Y3'), ('Y3', 'Y1')]
    
    for ax, (i, j), (lbl_i, lbl_j) in zip(axes, pairs, labels):
        # Plot real attractor in the 2D plane
        ax.plot(Y_true[:, i], Y_true[:, j], 
                color='black', alpha=0.5, linewidth=0.6, label='Real Attractor')
        
        # Plot closed-loop simulation
        ax.plot(mean_y[:, i], mean_y[:, j], 
                color='red', alpha=0.7, linewidth=0.6, label='Closed Loop')
        
        ax.set_xlabel(lbl_i)
        ax.set_ylabel(lbl_j)
        ax.set_title(f'Phase Plane: {lbl_i} vs {lbl_j}')
        ax.legend(loc='upper right')
        ax.grid(True)
        
    plt.suptitle('Attractor Comparison in 2D Projections', fontsize=14)
    plt.tight_layout()
    plt.savefig('media/closed_loop_attractor_2d.png', dpi=300)
    print("Plots saved to media/closed_loop_attractor_2d.png!")
    plt.show()

if __name__ == '__main__':
    plot_attractor_2d_projections()