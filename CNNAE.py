# Author: Holly Champion
# Year: 2025
# Title: CNN Autoencoder
# Description: Detects anomalies by a 1D-CNN autoencoder reconstruction error on normal waveforms cycles.

import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Constants
SEED = 42
# BATCH_SIZE = 256
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-6
EPOCHS = 200
PATIENCE = 20
LR = 5e-4
SAVE_BEST = "cnnAE_best.pt"
DEVICE = "cpu"

# Load Data
base = input("Enter person name: ").strip()
normal  = np.load(f"OriginalData/{base}_normal_autoencoder_ready.npy").astype(np.float32)
compress_50 = np.load(f"OriginalData/{base}_50_autoencoder_ready.npy").astype(np.float32)
compress_80 = np.load(f"OriginalData/{base}_80_autoencoder_ready.npy").astype(np.float32)

# Reproducability
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
set_seed()

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Split data into training pool (80%) and testing (20%)
normal_train_pool, normal_test = train_test_split(normal, test_size=0.2, random_state=42, shuffle=True)

# Split training pool into training (80%) and validation (20%)
normal_train, normal_validate = train_test_split(normal_train_pool, test_size=0.2, random_state=42, shuffle=True)

# Add a channel dimension and convert numpy array into pytorch tensor
normal_train_pool = torch.tensor(normal_train_pool[:, None, :], dtype=torch.float32)
normal_train = torch.tensor(normal_train[:, None, :], dtype=torch.float32)
normal_validate = torch.tensor(normal_validate[:, None, :], dtype=torch.float32)
normal_test = torch.tensor(normal_test[:, None, :], dtype=torch.float32)
compress_50 = torch.tensor(compress_50[:, None, :], dtype=torch.float32)
compress_80 = torch.tensor(compress_80[:, None, :], dtype=torch.float32)

#################### Model ####################

# 1D CNN autoencoder: T -> T/2 -> T/4 -> T/2 -> T"""
class CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (compress signal)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3, stride=2), nn.ReLU(),   # T -> T/2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2), nn.ReLU(),  # T/2 -> T/4
        )
        # Decoder (rebuild signal)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU(),  # T/4 -> T/2
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1),             # T/2 -> T
        )

    # Encode then decode
    def forward(self, x):
        return self.decoder(self.encoder(x))

#################### Training Setup ####################

# Create model in device
model = CNNAE().to(DEVICE)

# Adam optimiser (updates model weight during training)
optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Mean squared error loss without a reduction
mse = nn.MSELoss(reduction="none")

# Split dataset x into batches
def batches(X, batch_size):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end]

#################### Train ####################
best_val = float("inf")
wait = 0

# Number of samples
n_train = len(normal_train)
n_validate = len(normal_validate)

# Move data to device
normal_train = normal_train.to(DEVICE)
normal_validate  = normal_validate.to(DEVICE)

for epoch in range(1, EPOCHS + 1):

    # Training Data
    model.train()
    running_loss = 0.0

    for batch in batches(normal_train, BATCH_SIZE):
        optimiser.zero_grad()
        recon = model(batch)
        loss = mse(recon, batch).mean()

        loss.backward()
        optimiser.step()

        running_loss += loss.item() * len(batch)
    training_loss = running_loss / max(1, n_train)
    
    # Testing Data
    model.eval()
    running_loss = 0.0

    # Don't use a gradient so model doesn't get trained while testing
    with torch.no_grad():
        for batch in batches(normal_validate, BATCH_SIZE):
            recon = model(batch)
            loss = mse(recon, batch).mean()
            running_loss += loss.item() * len(batch)
    validation_loss = running_loss / max(1, n_validate)

    # Early stopping logic (if testing loss has not improved)
    if validation_loss < best_val - 1e-6:
        best_val = validation_loss
        torch.save(model.state_dict(), SAVE_BEST)
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train {training_loss:.6f} | Val {validation_loss:.6f}")

#################### Evaluate ####################

# Load the best saved model and put in evaluation mode
model.load_state_dict(torch.load(SAVE_BEST, map_location=DEVICE))
model.eval()

# Disable gradient tracking during evaluation 
@torch.no_grad()

# Compute reconstruction error
def recon_error(X):
    # Convert numpy array to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    errors = []
    for batch in batches(X_tensor, BATCH_SIZE):
        recon_batch = model(batch)
        batch_error = mse(recon_batch, batch).mean(dim=(1,2))
        errors.append(batch_error.cpu().numpy())
    return np.concatenate(errors)

# Compute reconstruction error for each dataset
err_pool   = recon_error(normal_train_pool.cpu().numpy())
err_train  = recon_error(normal_train.cpu().numpy())
err_val    = recon_error(normal_validate.cpu().numpy())
err_test   = recon_error(normal_test.cpu().numpy())
err_50     = recon_error(compress_50.cpu().numpy())
err_80     = recon_error(compress_80.cpu().numpy())

# 95th percentile threshold based on VALIDATION normals
threshold = np.percentile(err_val, 95)

# Ratio of samples above threshold (anomaly rate)
ratio_tr    = (err_train  > threshold).mean()
ratio_val   = (err_val    > threshold).mean()
ratio_test  = (err_test  > threshold).mean()
ratio_50    = (err_50     > threshold).mean()
ratio_80    = (err_80     > threshold).mean()

# Print results
print("\n 1D-CNN Autoencoder")
print(f"Normal (train): {ratio_tr:.2f}")
print(f"Normal (val):   {ratio_val:.2f}")
print(f"Normal (test):  {ratio_test:.2f}")
print(f"Comp 50%:       {ratio_50:.2f}")
print(f"Comp 80%:       {ratio_80:.2f}")