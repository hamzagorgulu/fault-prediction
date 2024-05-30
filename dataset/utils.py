import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

def read_dat_file(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data

def split_dataset(data, train_percent):
    total_len = len(data)
    train_len = int(total_len * train_percent)
    train = data.iloc[:train_len]
    test = data.iloc[train_len:]
    assert len(train) + len(test) == len(data), "Train and test split failed"
    return train, test

def create_sliding_windows(data, input_size, target_size, step_size, feature_cols, target_col, input_channels_first = True):
    X = []
    y = []
    #y_dates = []  # Add an empty list to store the dates
    print(f"Creating sliding windows with input_size={input_size}, target_size={target_size}, step_size={step_size}")
    for start in tqdm(range(0, len(data) - input_size - target_size + 1, step_size)):
        end = start + input_size
        target_end = end + target_size
        X.append(data.iloc[start:end][feature_cols].values)
        y.append(data.iloc[end:target_end][target_col].values)
        #y_dates.append(data.iloc[end:target_end]['current_datetime'].dt.date.values[-1])  # Store the dates
    
    X = np.array(X)
    y = np.array(y)
    #y_dates = np.array(y_dates)  # Convert the dates list to a numpy array
    #y_dates = np.unique(y_dates)  # Remove duplicate dates

    if input_channels_first:
        X = np.transpose(X, (0, 2, 1))  # Convert to (batch_size, input_channels, sequence_length)
    
    return X, y  # Return the dates along with X and y


def random_crop(x, crop_length=50):
    start = torch.randint(0, x.shape[2] - crop_length, (1,))
    return x[:, :, start:start + crop_length]

def add_noise(x, noise_level=0.05):
    noise = torch.randn_like(x) * noise_level
    return x + noise


def jitter(x, sigma): #0.08
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, loc, sigma): #0.1
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=loc, scale=sigma, size=(1, x.shape[1], 1))
 
    return x * factor

def permutation(x, max_segments):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1: 
        split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points)

        # print([s.shape for s in splits])  # Check the shapes of splits
        # permuted_splits = np.random.permutation(splits)
        # print([p.shape for p in permuted_splits])  # Check the shapes after permutation

        np.random.shuffle(splits) # shuffle the splits in place
        warp = np.concatenate(splits).ravel()

        

        return x[:, warp]
    else:
        return x