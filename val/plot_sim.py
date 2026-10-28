import torch
import scipy.io
from config import idx_0, N_train, N_test, device
import matplotlib.pyplot as plt

def plot():
    data = scipy.io.loadmat("NHLR_data.mat")
    U_raw = data["u"]
    
    X_test = U_raw[0:3][:, idx_0 + N_train : idx_0 + N_train + N_test]
    Y_test = U_raw[3:6][:, idx_0 + N_train : idx_0 + N_train + N_test]
    Z_test = U_raw[6:9][:, idx_0 + N_train : idx_0 + N_train + N_test]
    
    X_sim = torch.load("cfm_sim.pt", map_location=device)
    
    plt.subplot(3,3,1)
    plt.plot(X_sim[:,0])
    plt.plot(X_test[0], '--')
    plt.subplot(3,3,2)
    plt.plot(X_sim[:,1])
    plt.plot(X_test[1], '--')
    plt.subplot(3,3,3)
    plt.plot(X_sim[:,2])
    plt.plot(X_test[2], '--')
    
    plt.subplot(3,3,4)
    plt.plot(X_sim[:,3])
    plt.plot(Y_test[0], '--')
    plt.subplot(3,3,5)
    plt.plot(X_sim[:,4])
    plt.plot(Y_test[1], '--')
    plt.subplot(3,3,6)
    plt.plot(X_sim[:,5])
    plt.plot(Y_test[2], '--')
    
    plt.subplot(3,3,7)
    plt.plot(X_sim[:,6])
    plt.plot(Z_test[0], '--')
    plt.subplot(3,3,8)
    plt.plot(X_sim[:,7])
    plt.plot(Z_test[1], '--')
    plt.subplot(3,3,9)
    plt.plot(X_sim[:,8])
    plt.plot(Z_test[2], '--')
    plt.show()

if __name__ == "__main__":
    plot()