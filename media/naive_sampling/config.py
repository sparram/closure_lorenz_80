import torch

# GENERAL PARAMETERS
data_path = "data/NN_training_data.mat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATALOAD PARAMETERS
idx_0 = 100000
N_train = 5000000
N_test = 100000

# CFM PARAMETERS
K = 100           # Total memory points
stride = 50       # Memory stride
cfm_steps = 8    # CFM Flow inference steps
hidden_dim = 256   # CFM Hidden layer dimension

# TRAINING PARAMETERS
lr=1e-3
epochs = 20
batch_size = 1024

# A POSTERIORI SIMULATION PARAMETERS
dt = 4.2e-3 # 4.2e-3 MTU = 45 sec
N_ens = 50
sigma0 = 1e-3