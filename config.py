# GENERAL PARAMETERS
data_path = "data/NN_training_data.mat"
device = 'cpu'

# DATALOAD PARAMETERS
idx_0 = 100000
N_train = 4000000
N_test = 5000

# CFM PARAMETERS
K = 50           # Total memory points
stride = 100       # Memory stride
cfm_steps = 10    # CFM Flow inference steps
hidden_dim = 8   # CFM Hidden layer dimension

# TRAINING PARAMETERS
lr=1e-3
epochs = 5

# A POSTERIORI SIMULATION PARAMETERS
dt = 4.2e-3 # 4.2e-3 MTU = 45 sec
N_ens = 1024
sigma0 = 1e-3