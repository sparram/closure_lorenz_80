import torch
from src.cfm import CFM, FlowModel
from src.data_loader import get_dataloader
import tqdm
from config import lr, epochs, device

def train():
    train_loader = get_dataloader()
    
    v_flow = FlowModel()
    v_flow.to(device)
    cfm = CFM(v_flow)
    optimizer = torch.optim.Adam(v_flow.parameters(), lr=lr)
    
    for epoch in range(epochs):
        v_flow.train()
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch : {epoch+1}")
    
        train_loss = 0.0
        for input_batch, out_batch in progress_bar:
            input_batch = input_batch.to(device, dtype=torch.float32)
            out_batch = out_batch.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            loss = cfm.compute_loss(input_batch, out_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        print(f"Mean Epoch Loss: {train_loss}")

        if (epoch+1) % 5 == 0:
            torch.save(v_flow.state_dict(), f"cfm_model_{epoch+1}epochs.pt")
    print("TRAINING COMPLETED!")

if __name__ == "__main__":
    train()