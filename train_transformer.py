import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

from simclr.modules import NT_Xent

from models.simclr_transformer import SimCLR_Transformer
from dataset.utils import split_dataset, create_sliding_windows, jitter, scaling, permutation

def random_crop(x, crop_length=50):
    start = torch.randint(0, x.shape[2] - crop_length, (1,))
    return x[:, :, start:start + crop_length]

def add_noise(x, noise_level=0.05):
    noise = torch.randn_like(x) * noise_level
    return x + noise


df = pd.read_csv("data/dataset_small_together.csv")

input_size = 50
target_size = 1
step_size = 1
split_rate = 1.0

train, test = split_dataset(df, split_rate)

feature_cols = ['xmeas_1', 'xmeas_2', 'xmeas_3',
       'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9',
       'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15',
       'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21',
       'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27',
       'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32', 'xmeas_33',
       'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39',
       'xmeas_40', 'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5',
       'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']
target_col = 'fault'

window_file = f"data/windows/input_{input_size}_target_{target_size}_step_{step_size}_split_{split_rate}"

if os.path.exists(window_file):
    # Load the preprocessed sliding windows
    X_train = np.load(os.path.join(window_file, f"X_train.npy"))
    y_train = np.load(os.path.join(window_file, f"y_train.npy"))
else:
    # Create the sliding windows
    X_train, y_train = create_sliding_windows(train, input_size, target_size, step_size, feature_cols, target_col, input_channels_first=True)

    # create the directory if it does not exist
    os.makedirs(window_file, exist_ok=True)

    # Save the sliding windows
    np.save(os.path.join(window_file, f"X_train.npy"), X_train)
    np.save(os.path.join(window_file, f"y_train.npy"), y_train)

# convert to torch tensor
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# Create a DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_channels = 52 # number of features in the input data
sequence_length = input_size
batch_size = 32
epochs = 10
learning_rate = 0.001
feature_dim = 128
temperature = 0.1

# Model
model = SimCLR_Transformer(input_channels, feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = NT_Xent(batch_size, temperature, world_size=1)

# Dummy dataset
# x_train = torch.randn(2000, input_channels, sequence_length).to(device)  # Random data)
# train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=False)
breakpoint()
# Training loop
for epoch in range(epochs):
    for data in train_loader:
        x = data[0].to(device) # (batch_size, input_channels, sequence_length)

        # scaling and jittering as weak augmentation
        x_i = jitter(scaling(x, loc=1.8, sigma=0.1), sigma=0.08).type(torch.FloatTensor)

        # permutation and scaling as strong augmentation
        x_j = scaling(permutation(x, max_segments=17), loc=0.5, sigma=0.1).type(torch.FloatTensor)

        optimizer.zero_grad()
        z_i = model(x_i)
        z_j = model(x_j)
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        print(loss)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
