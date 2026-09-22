# PROJECT STRUCTURE

```text
closure_lorenz_80/
├── checkpoints/
├── data/
│   ├── NHLR_data.mat                         # Slow dynamics generated dataset (NO HLR)
│   └── NN_training_data.mat                  # Original training data (HLR)
├── docs/
├── media/
├── notebooks/
│   ├── l80_model.ipynb                       # Loading NN_training_data and exploring the physics
│   └── sim_lorenz_80.ipynb                   # Complete Implemnetation of the Lorenz-80 model
├── src/
│   ├── __init__.py
│   ├── physics.py                          # System physics of the L80 Model (only for Y)
│   ├── load_data.py                        # Data loading and transient filtering
│   ├── model.py                            # Conditional Velocity Field architecture (NN)
│   └── flow_matching.py                    # OT-FM System (Loss, solving the ODE to sample)
├── val/
│   ├── closed_loop_dynamics.py             # Closed-loop evolution of Y, X and Z
│   ├── open_loop_attractor.py              # Open-loop attractor of X, Z conditionally to Y
│   ├── open_loop_dynamics.py               # Open-loop dynamics of X, Z given Y
│   └── physics_validator.py                # Physics validation of physics.py code
├── train.py                                # Training script
└── requirements.txt
```