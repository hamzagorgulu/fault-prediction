import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm
from dataset.utils import split_dataset, create_sliding_windows, jitter, scaling, permutation
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.simclr_cnn import SimCLR_TS

# Set environment variables to prevent OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

def load_data(df_path, input_size, target_size, step_size, split_rate, feature_cols, target_col):
    df = pd.read_csv(df_path)
    train, test = split_dataset(df, split_rate)
    
    window_name = f"input_{input_size}_target_{target_size}_step_{step_size}_split_{split_rate}"
    window_path = os.path.join("data", "windows", window_name)
    
    if os.path.exists(window_path):
        X_train = np.load(os.path.join(window_path, f"X_train.npy"))
        y_train = np.load(os.path.join(window_path, f"y_train.npy"))
    else:
        X_train, y_train = create_sliding_windows(train, input_size, target_size, step_size, feature_cols, target_col, input_channels_first=True)
        os.makedirs(window_path, exist_ok=True)
        np.save(os.path.join(window_path, f"X_train.npy"), X_train)
        np.save(os.path.join(window_path, f"y_train.npy"), y_train)
    
    return X_train, y_train, window_name

def create_dataloader(X_train, y_train, batch_size):
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def extract_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            x = data[0].to(device)
            feature = model(x)
            features.append(feature.cpu().numpy())
    return np.concatenate(features)

def visualize_tsne(df_path, model_path, input_size=50, target_size=1, step_size=1, split_rate=1, batch_size=32, feature_dim=128, epoch_loss=0.17858992459080952):
    feature_cols = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10',
                    'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
                    'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30',
                    'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40',
                    'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']
    target_col = 'fault'

    X_train, y_train, window_name = load_data(df_path, input_size, target_size, step_size, split_rate, feature_cols, target_col)
    train_loader = create_dataloader(X_train, y_train, batch_size)

    y_train = torch.from_numpy(y_train).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = 52

    model = SimCLR_TS(input_channels, feature_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    train_features = extract_features(train_loader, model, device)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, verbose=True, n_jobs= -1)
    train_features_2d = tsne.fit_transform(train_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(train_features_2d[:, 0], train_features_2d[:, 1], c=y_train[:train_features_2d.shape[0]].to(torch.int32), cmap='viridis')
    plt.legend(handles=scatter.legend_elements()[0], labels=set(y_train[:train_features_2d.shape[0]]))
    plt.title('t-SNE visualization of the features')
    plt.savefig(f"figures/tsne_{window_name}.png")

# Example usage:
df_path = "data/dataset_small_together.csv"
epoch_loss = 0.17858992459080952
model_path = f"checkpoints/simclr-cnn_loss:{epoch_loss}_input_50_target_1_step_1_split_1.pth"
visualize_tsne(df_path, model_path)
breakpoint()