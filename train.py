from src.load_data import get_dataloaders


train_loader, val_loader = get_dataloaders('data/NN_training_data.mat', batch_size=256)
for y_batch, xz_batch in train_loader:
    # y_batch shape: torch.Size([256, 3])
    print(y_batch.shape)
    # fast_batch shape: torch.Size([256, 6])
    print(xz_batch.shape)
    break