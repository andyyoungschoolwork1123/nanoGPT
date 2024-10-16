# Import necessary libraries
import pandas as pd
import numpy as np
import os
from tqdm import tqdm 
from tstok import generic  # Import Config class from generic.py
from tstok.tokenizer import Tokenizer  # Import Tokenizer class from tokenizer.py
import torch

# Load the CSV file
csv_file_path = 'C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\data\\clean\\IBM'


def load_and_concatenate_csvs(folder_path):
    # Get a list of all CSV files in the directory and its subdirectories
    csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith('.csv')
    ]
    
    # List to store dataframes for concatenation
    all_dataframes = []

    # Loop through CSV files and load them into dataframes, with a progress bar
    for file in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(file, header=0)
        
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase for consistency
        rename_mapping = {
            'date': 'date',
            'timestamp': 'date',  # Assuming some files use 'timestamp' instead of 'date'
            'close_price': 'close',  # Assuming some files use 'close_price' instead of 'close'
        }
        
        # Rename columns to standard names if they exist in the dataframe
        df.rename(columns=rename_mapping, inplace=True)
        
        all_dataframes.append(df)
    
    # Concatenate all dataframes using Pandas
    all_data = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort the data by the 'date' column
    sorted_data = all_data.sort_values(by='date')

    # Save the sorted data to 'alldata.csv'
    sorted_data.to_csv(os.path.join(os.path.dirname(__file__), 'alldata.csv'), index=False)
    
    return sorted_data

class CustomDataset:
    '''copy from tstok-data.py'''
    def __init__(self,
                 series: list[np.array],
                 tokenizer: Tokenizer,
                 cfg: generic.Config):
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
            # randomly select a series
            series_ix = np.random.randint(len(self.tr_series))
            # randomly select a subsequence
            ts_start_ix = np.random.randint(0, self.tr_lengths[series_ix] - self.max_seq_len)
        else:
            series_ix = np.random.randint(len(self.val_series))
            ts_start_ix = np.random.randint(0, self.val_lengths[series_ix] - self.max_seq_len)

        sequence = self.series[series_ix][ts_start_ix:ts_start_ix+self.max_seq_len+1]
        return sequence[:-1], sequence[1:]
    
    def _norm_and_tokenize(self, Xs, Ys):
        '''
        Xs.shape => batch x seq_len
        Ys.shape => batch x seq_len
        '''
        # compute params that will be used as standardizing param for successive targets.
        hindsight_means = np.zeros(Xs.shape)
        hindsight_std = np.zeros(Ys.shape)
        for i in range(Xs.shape[1]):
            hindsight_means[:, i] = Xs[:, :i+1].mean(axis=1)
            hindsight_std[:, i] = Xs[:, :i+1].std(axis=1)

        # standardize the context windows using it's own mean and std
        Xs_std = (Xs - Xs.mean(axis=1).reshape(-1, 1)) / (Xs.std(axis=1).reshape(-1, 1) + 1e-6)

        # standardize the targets using the context windows' lagging mean and std
        Ys_std = (Ys - hindsight_means) / (hindsight_std + 1e-6)
        
        X_ids = self.tokenizer.digitize(Xs_std)
        Y_ids = self.tokenizer.digitize(Ys_std)

        return X_ids, Y_ids

    def get_batch(self, batch_size, split):
        Xs = []; Ys = []
        for _ in range(batch_size):
            X, Y = self._get_sample(split)
            Xs.append(X)
            Ys.append(Y)
        
        Xs = np.stack(Xs); Ys = np.stack(Ys)
        X_ids, Y_ids = self._norm_and_tokenize(Xs, Ys)

        X_ids = torch.from_numpy(X_ids).to(torch.long)
        Y_ids = torch.from_numpy(Y_ids).to(torch.long)

        if self.device == 'cuda':
            # pin arrays x,y
            # which allows us to move them to GPU asynchronously (non_blocking=True)
            X_ids = X_ids.pin_memory().to(self. device, non_blocking=True)
            Y_ids = Y_ids.pin_memory().to(self.device, non_blocking=True)
        # else:
        #     x, y = x.to(self.device), y.to(self.device)
        return X_ids, Y_ids

# Function to round values to four significant figures
# Function to round values to four significant figures using progress bar
def round_to_4sf_with_progress(values):
    total = len(values)
    rounded_values = []

def prepare_data(csv_path, tokenizer, cfg):
    # Load your Tesla data CSV file
    df = pd.read_csv(csv_path)

    # Select the relevant columns for time-series analysis
    # In this case, we're using the 'open', 'high', 'low', 'close', and 'volume' columns
    columns_to_use = ['open', 'high', 'low', 'close', 'volume']
    series = [df[col].values for col in columns_to_use]

    # Convert the list of series to the format expected by CustomDataset
    dataset = CustomDataset(series, tokenizer, cfg)

    return dataset

data_config = {
    # dataloader
    "max_seq_len": 256, # 512 + 1 for the target
    
    # if gradient_accumulation_steps > 1, this is the micro-batch size
    # This is same as the max_seq_len in data_config
    "batch_size": 64,
    # tokenizer
    "bin_size": 0.005,
    "max_coverage": .9998,
    "train_ratio": 0.95
}
training_config = {
    # adamw optimizer
    "learning_rate": 6e-4,              # max learning rate
    "max_iters": 1000,                # total number of training iterations
    "weight_decay": 4e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,                   # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    "decay_lr": True,                   # whether to decay the learning rate
    "warmup_iters": 50,                # how many steps to warm up for
    "lr_decay_iters": 1000,           # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # training opts
    "grad_accu_steps": 2 , # used to simulate larger batch sizes
    
    # system - training opts
    "device": "cpu", # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "dtype": "float32", # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    "compile": False, # use PyTorch 2.0 to compile the model to be faster
}

# Example usage
csv_path = "data\\clean\\IBM\\alldata.csv"
cfg = generic.Config() 
tokenizer = Tokenizer()  # You need to provide the appropriate arguments

dataset = prepare_data(csv_path, tokenizer, cfg)



