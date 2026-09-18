# PROJECT PLAN

```text
l80_neural_closure/
├── data/
│   └── NN_training_data.mat       # Datos crudos de simulación
├── configs/
│   └── config.yaml                # Hiperparámetros (lr, batch_size, dt, steps_tau)
├── src/
│   ├── __init__.py
│   ├── physics.py                 # Ecuaciones reales de L80 (física del sistema)
│   ├── load_data.py                 # Carga de datos y filtrado de transitorios (t < 10)
│   ├── model.py                  # Arquitectura del Conditional Velocity Field
│   ├── flow_matching.py           # Trainer OT-FM (Loss, muestreo sintético en tau)
├── train.py                       # Script ejecutable para entrenamiento offline
├── simulate_online.py             # Script ejecutable para simulación online y gráficos
└── requirements.txt
```