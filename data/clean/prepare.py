import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import tiktoken
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

    # Loop through CSV files and load them into dataframes
    for file in csv_files:
        df = pd.read_csv(file, header=0)
        
        df.columns = df.columns.str.lower()  # Convert all column names to lowercase for consistency
        rename_mapping = {
            'date': 'date',
            'timestamp': 'date',  # Assuming some files use 'timestamp' instead of 'date'
            'data': 'close',  # Assuming some files use 'close_price' instead of 'close'
        }
        
        # Rename columns to standard names if they exist in the dataframe
        df.rename(columns=rename_mapping, inplace=True)
        
        
        all_dataframes.append(df)
    
    # Concatenate all dataframes using Pandas
    all_data = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort the data by the 'date' column
    sorted_data = all_data.sort_values(by='date')

    # Save the sorted data to 'alldata.csv'
    #sorted_data.to_csv('alldata.csv', index=False)
    
    return sorted_data

# Define the function to process the date and price into tokens
def process_date_and_price_to_tokens_plain(df):
    
    # Extract and tokenize the date
    dates = df['date']
    
    # Convert the list of characters in date strings to tokens (ASCII codes)
    date_tokens = dates.apply(lambda x: [ord(c) for c in str(x)])
    
    # Extract the stock prices and convert them to tokens
    prices = df['close'].astype(str).apply(lambda x: list(x.replace(' ', '')))
    price_tokens = prices.apply(lambda x: [ord(char) for char in x])
    
    # Flatten the list of tokens for both dates and prices into a single list of tokens
    all_tokens = []
    for date, price in zip(date_tokens, price_tokens):
        all_tokens.extend(date)  # Add date tokens
        all_tokens.append(ord(','))  # Add a separator between date and price (e.g., comma)
        all_tokens.extend(price)  # Add price tokens
        all_tokens.append(ord('\n'))  # Add a newline token after each entry
    
    return np.array(all_tokens, dtype=np.uint8)


def process_date_and_price_to_tokens(df):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Extract date and convert to string format
    dates = df['date'].astype(str)
    # Tokenize dates
    date_tokens = dates.apply(lambda x: tokenizer.encode(x))

    # Extract stock prices and convert them to string
    prices = df['close'].astype(str)
    # Tokenize prices
    price_tokens = prices.apply(lambda x: tokenizer.encode(x))

    # Flatten the list of tokens for both dates and prices into a single list of tokens
    all_tokens = []
    for date, price in zip(date_tokens, price_tokens):
        all_tokens.extend(date)  # Add date tokens
        all_tokens.extend(tokenizer.encode(','))  # Add a separator between date and price (e.g., comma)
        all_tokens.extend(price)  # Add price tokens
        all_tokens.extend(tokenizer.encode('\n'))  # Add a newline token after each entry

    return np.array(all_tokens, dtype=np.uint16)  # Use uint16 to support larger token values



# Define the function to save the token data to binary files
def save_to_bin_file(data, filename):
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

# Main function to prepare the data and save train.bin and val.bin
def prepare_data():
    # Step 1: Load and concatenate CSV files
    data = load_and_concatenate_csvs(CSV_FOLDER)
    
    # Step 2: Process the stock price data into tokens
    tokens = process_date_and_price_to_tokens (data)
    
    # Step 3: Split the tokens into training and validation sets
    train_size = int(len(tokens) * TRAIN_RATIO)
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    # Step 4: Save the tokens to binary files
    
    train_tokens.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_tokens.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    print(f"Data preparation completed. Train size: {len(train_tokens)}, Validation size: {len(val_tokens)}")

if __name__ == '__main__':
    prepare_data()
    


