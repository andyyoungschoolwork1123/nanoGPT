import os
import numpy as np
import pandas as pd
import tiktoken
import dask.dataframe as dd
from tqdm import tqdm  # Import tqdm for progress bar

# Specify the folder containing the CSV files
CSV_FOLDER = 'C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\data\\clean'

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

    # Loop through CSV files and load them into dataframes, with a progress bar
    for file in tqdm(csv_files, desc="Loading CSV files"):
        df = pd.read_csv(file, header=0)
        
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase for consistency
        rename_mapping = {
            'date': 'date',
            'timestamp': 'date',  # Assuming some files use 'timestamp' instead of 'date'
            'close_price': 'close'  # Assuming some files use 'close_price' instead of 'close'
        }
        
        # Rename columns to standard names if they exist in the dataframe
        df = df.rename(columns=rename_mapping)
        
        # Only select the required columns
        if 'date' in df.columns and 'close' in df.columns:
            df = df[['date', 'close','symbol']]
            df = df.reset_index(drop=True)  # Reset index to avoid duplicate index values
            all_dataframes.append(df)
    
    # Concatenate all dataframes using dask
    all_data = dd.concat(all_dataframes, interleave_partitions=True)

    # Save to CSV if needed
    all_data.to_csv('alldata.csv', index=False, single_file=True)  # Writing to a single CSV file with Dask
    
    return all_data

load_and_concatenate_csvs(CSV_FOLDER)


