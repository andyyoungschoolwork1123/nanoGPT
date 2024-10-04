import os
import numpy as np
import pandas as pd
import tiktoken
import dask.dataframe as dd
from tqdm import tqdm  # Import tqdm for progress bar
from dask.diagnostics import ProgressBar
import gc

# Specify the folder containing the CSV files
CSV_FOLDER = 'C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\data\\clean'

TRAIN_RATIO = 0.8  # 80% of the data will be used for training, 20% for validation




def process_all_to_tokens(csv):
    df = dd.read_csv(csv)
    
    df = df.repartition(npartitions=8)

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Add progress bars for tokenizing the columns
    tqdm.pandas(desc="Tokenizing date")
    df['date_tokens'] = df['date'].astype(str).map(lambda x: tokenizer.encode(x))

    tqdm.pandas(desc="Tokenizing price")
    df['price_tokens'] = df['close'].astype(str).map(lambda x: tokenizer.encode(x))

    if 'symbol' in df.columns:
        tqdm.pandas(desc="Tokenizing symbol")
        df['symbol_tokens'] = df['symbol'].astype(str).map(lambda x: tokenizer.encode(x))

    gc.collect()
    
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

    
    df['combined_tokens'] = df.apply(combine_tokens, axis=1, meta=('combined_tokens', 'object'))

        # Flatten the token lists into a single list using
    all_tokens = []
    for tokens in tqdm(df['combined_tokens'], desc="Flattening tokens"):
        all_tokens.extend(tokens)

    return np.array(all_tokens, dtype=np.uint16)

def main():
    
    tokens = process_all_to_tokens("C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\alldata.csv")
    train_size = int(len(tokens) * TRAIN_RATIO)
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    # Step 4: Save the tokens to binary files
    train_tokens.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_tokens.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    print(f"Data preparation completed. Train size: {len(train_tokens)}, Validation size: {len(val_tokens)}")

if __name__ == "__main__":
    main()



