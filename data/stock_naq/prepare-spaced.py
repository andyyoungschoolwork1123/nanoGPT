# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import pandas as pd
from datasets import load_dataset
import torch
 #CSV_FOLDER = 'C:/Users/catop/Documents/GitHub/nanoGPT/data/stock_dat/clean'


TRAIN_RATIO = 0.8  # 80% of the data will be used for training, 20% for validation

# Define the function to load all CSVs and concatenate them into a single DataFrame
def load_and_concatenate_csvs(folder_path):
    # Get a list of all CSV files in the directory and its subdirectories
    csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith('.csv')
    ]
    
    # List to store dataframes for concatenation
    all_dataframes = []

    # Loop through CSV files and load them into dataframes
    for file in csv_files:
        df = pd.read_csv(file, header=0)
        
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase for consistency
        rename_mapping = {
            'd a t a': 'date',
            'c l o s e': 'close',  # Assuming some files use 'timestamp' instead of 'date'
            'd a t e': 'date',
            "t i m e s t a m p" : 'date'# Assuming some files use 'close_price' instead of 'close'
        }
        
        # Rename columns to standard names if they exist in the dataframe
        df.rename(columns=rename_mapping, inplace=True)
        
        
        all_dataframes.append(df)
    
    # Concatenate all dataframes using Pandas
    all_data = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort the data by the 'date' column

    # Save the sorted data to 'alldata.csv'
    all_data.to_csv('alldata-spaced.csv', index=False)
    return all_data

class CustomDataset:
    '''Custom dataset for time-series data.'''
    def __init__(self,
                 series: list[np.array],
                 tokenizer,
                 cfg):
        # each element in series is a 1-array of time-series values
        self.series = series
        self.tr_series = series[:int(len(series) * cfg.data.train_ratio)]
        self.val_series = series[int(len(series) * cfg.data.train_ratio):]

        self.tr_lengths = [len(s) for s in self.tr_series]
        self.val_lengths = [len(s) for s in self.val_series]
        
        self.max_seq_len = cfg.data.max_seq_len
        self.train_ratio = cfg.data.train_ratio
        self.device = cfg.training.device
        self.tokenizer = tokenizer

    def _get_sample(self, split):
        if split == 'train':
            while True:  # Retry until a valid series is found
                series_ix = np.random.randint(len(self.tr_series))
                if self.tr_lengths[series_ix] > self.max_seq_len:
                    break
            
            ts_start_ix = np.random.randint(0, self.tr_lengths[series_ix] - self.max_seq_len)
        else:
            while True:  # Retry until a valid series is found
                series_ix = np.random.randint(len(self.val_series))
                if self.val_lengths[series_ix] > self.max_seq_len:
                    break
            
            ts_start_ix = np.random.randint(0, self.val_lengths[series_ix] - self.max_seq_len)

        sequence = self.series[series_ix][ts_start_ix:ts_start_ix+self.max_seq_len+1]
        return sequence[:-1], sequence[1:]

    def _norm_and_tokenize(self, Xs, Ys):
        '''
        Xs.shape => batch x seq_len
        Ys.shape => batch x seq_len
        '''
        # Compute params used as standardizing params for successive targets
        hindsight_means = np.zeros(Xs.shape)
        hindsight_std = np.zeros(Ys.shape)
        for i in range(Xs.shape[1]):
            hindsight_means[:, i] = Xs[:, :i+1].mean(axis=1)
            hindsight_std[:, i] = Xs[:, :i+1].std(axis=1)

        # Standardize the context windows using its own mean and std
        Xs_std = (Xs - Xs.mean(axis=1).reshape(-1, 1)) / (Xs.std(axis=1).reshape(-1, 1) + 1e-6)

        # Standardize the targets using the context windows' lagging mean and std
        Ys_std = (Ys - hindsight_means) / (hindsight_std + 1e-6)
        
        # Tokenize (digitize) the standardized data
        X_ids = self.tokenizer.digitize(Xs_std)
        Y_ids = self.tokenizer.digitize(Ys_std)

        return X_ids, Y_ids

    def get_batch(self, batch_size, split):
        Xs = []; Ys = []
        for _ in range(batch_size):
            X, Y = self._get_sample(split)
            Xs.append(X)
            Ys.append(Y)
        
        Xs = np.stack(Xs)
        Ys = np.stack(Ys)
        X_ids, Y_ids = self._norm_and_tokenize(Xs, Ys)

        X_ids = torch.from_numpy(X_ids).to(torch.long)
        Y_ids = torch.from_numpy(Y_ids).to(torch.long)

        if self.device == 'cuda':
            # Pin memory for async data transfer to GPU
            X_ids = X_ids.pin_memory().to(self.device, non_blocking=True)
            Y_ids = Y_ids.pin_memory().to(self.device, non_blocking=True)

        return X_ids, Y_ids
    
    def get_all_tokens(self, split='train'):
        '''
        Get all tokenized sequences from the dataset for a specific split (train/val).
        '''
        Xs = []; Ys = []
        for i in range(len(self.tr_series)):
            X, Y = self._get_sample(split)
            Xs.append(X)
            Ys.append(Y)
        
        Xs = np.stack(Xs); Ys = np.stack(Ys)
        X_ids, Y_ids = self._norm_and_tokenize(Xs, Ys)
        
        return np.concatenate([X_ids.flatten(), Y_ids.flatten()])

# Function to round values to four significant figures with progress bar
def round_to_4sf_with_progress(values):
    total = len(values)
    rounded_values = []
    for value in tqdm(values, total=total, desc="Rounding values"):
        rounded_values.append(round(value, 4))
    return rounded_values

# Prepare the dataset
def prepare_data(csv_path, tokenizer, cfg):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Select relevant columns for time-series analysis
    columns_to_use = ['date', 'close']

    # Convert 'close' column to numeric (if needed)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')  # Convert 'close' column to float, set invalid parsing as NaN
    
    # Drop any rows with NaN values after conversion
    df.dropna(subset=['close'], inplace=True)

    # Convert the relevant columns into numpy arrays
    series = [df[col].values for col in columns_to_use]

    # Create and return the dataset
    dataset = CustomDataset(series, tokenizer, cfg)
    return dataset


def write_tokens_to_bin(tokenized_data, file_path):
    '''
    Writes tokenized data to a binary file.
    
    Args:
        tokenized_data: numpy array of token IDs
        file_path: path to the .bin file
    '''
    dtype = np.uint16  # Assuming the token values will fit into uint16
    arr_len = tokenized_data.shape[0]
    
    # Use memory-mapped file to write the data
    arr = np.memmap(file_path, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = tokenized_data[:]  # Copy tokenized data
    arr.flush()  # Ensure it's written to disk
    

# Example data and configuration
data_config = {
    "max_seq_len": 256,
    "batch_size": 32,
    "bin_size": 0.005,
    "max_coverage": .9998,
    "train_ratio": 0.80
}

training_config = {
    "learning_rate": 6e-4,
    "max_iters": 1000,
    "weight_decay": 4e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 50,
    "lr_decay_iters": 1000,
    "min_lr": 6e-5,
    "grad_accu_steps": 2,
    "device": "cpu",
    "dtype": "float32",
    "compile": False
}
import tstok.configurations
import tstok.generic as generic
from tstok.tokenizer import Tokenizer
# Assume a generic Config class and Tokenizer class are defined elsewhere
cfg = generic.Config(config = tstok.configurations.all_config)
tokenizer = Tokenizer(cfg.data)

csv_path = "data\\stock_naq\\alldata.csv"
dataset = prepare_data(csv_path, tokenizer, cfg)

tokenized_data = dataset.get_all_tokens(split='train')
    
    # Define the path to save the binary file
train_bin_path = os.path.join(os.getcwd(), 'train.bin')
    
    # Write tokens to binary file
write_tokens_to_bin(tokenized_data, train_bin_path)
    
print(f"Tokenized data written to {train_bin_path}")
    
