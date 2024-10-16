# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import pandas as pd
from datasets import load_dataset

CSV_FOLDER = 'C:/Users/catop/Documents/GitHub/nanoGPT/data/clean'
TRAIN_RATIO = 0.8  # 80% of the data will be used for training, 20% for validation

# Function to load and concatenate CSVs
def load_and_concatenate_csvs(folder_path):
    csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith('.csv')
    ]

    all_dataframes = []
    for file in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(file, header=0)
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase
        rename_mapping = {
            'date': 'date',
            'timestamp': 'date',  # Rename if timestamp is used
            'close_price': 'close'  # Ensure 'close' column is standardized
        }
        df = df.rename(columns=rename_mapping)
        if 'date' in df.columns and 'close' in df.columns:
            df = df[['date', 'close', 'symbol']]  # Keep relevant columns
            df = df.reset_index(drop=True)
            all_dataframes.append(df)

    all_data = pd.concat(all_dataframes, ignore_index=True)
    all_data.sort_values(by='date', inplace=True)
    all_data.to_csv('alldata-lesscol.csv', index=False)
    return all_data

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load and concatenate CSVs
    csv = load_and_concatenate_csvs(CSV_FOLDER)
    dataset = load_dataset("csv", data_files='alldata-lesscol.csv', num_proc=num_proc_load_dataset)

    # Split the dataset into train/val sets
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename 'test' split to 'val'

    # Define the processing function for tokenization
    def process(example):
        # Encode the 'close' field as a string since 'close' is numerical
        text = str(example['close'])
        ids = enc.encode_ordinary(text)  # Encode the 'close' value as text
        ids.append(enc.eot_token)  # Add end of text token
        return {'ids': ids, 'len': len(ids)}

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['close'],  # Remove 'close' after encoding
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Write tokenized data to binary format
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # Use uint16 as the dtype
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster writing
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
    
