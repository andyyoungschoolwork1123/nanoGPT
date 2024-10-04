import os
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm  # Import tqdm for progress bar

# Specify the folder containing the CSV files
CSV_FOLDER = 'C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\data\\clean\\IBM'

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

def process_all_to_tokens(df):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Add progress bars for tokenizing the columns
    tqdm.pandas(desc="Tokenizing date")
    df['date_tokens'] = df['date'].astype(str).progress_apply(lambda x: tokenizer.encode(x))

    tqdm.pandas(desc="Tokenizing price")
    df['price_tokens'] = df['close'].astype(str).progress_apply(lambda x: tokenizer.encode(x))

    if 'symbol' in df.columns:
        tqdm.pandas(desc="Tokenizing symbol")
        df['symbol_tokens'] = df['symbol'].astype(str).progress_apply(lambda x: tokenizer.encode(x))

    # Combine all tokens into a single list for each row
    def combine_tokens(row):
        combined_tokens = row['date_tokens']
        combined_tokens += tokenizer.encode(',')
        combined_tokens += row['price_tokens']
        combined_tokens += tokenizer.encode(',')
        if 'symbol_tokens' in row:
            combined_tokens += row['symbol_tokens']
        combined_tokens += tokenizer.encode('\n')
        return combined_tokens

    tqdm.pandas(desc="Combining tokens")
    df['combined_tokens'] = df.progress_apply(combine_tokens, axis=1)

    # Flatten the token lists into a single list
    all_tokens = []
    for tokens in tqdm(df['combined_tokens'], desc="Flattening tokens"):
        all_tokens.extend(tokens)

    return np.array(all_tokens, dtype=np.uint16)

def main():
    data = load_and_concatenate_csvs(CSV_FOLDER)
    tokens = process_all_to_tokens(data)
    train_size = int(len(tokens) * TRAIN_RATIO)
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    # Step 4: Save the tokens to binary files
    train_tokens.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_tokens.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    print(f"Data preparation completed. Train size: {len(train_tokens)}, Validation size: {len(val_tokens)}")

main()


