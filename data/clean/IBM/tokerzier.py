# Import necessary libraries
import pandas as pd
import numpy as np
import os
from tqdm import tqdm 
from tstok import generic  # Import Config class from generic.py
from tstok.tokenizer import Tokenizer  # Import Tokenizer class from tokenizer.py

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

data = load_and_concatenate_csvs(csv_file_path)

# Define a configuration dictionary for Tokenizer
config_data = {
    "bin_size": 100,  # Example bin size, can be adjusted
    "max_coverage": 5  # Standard deviations to cover in Gaussian bins
}

# Create a Config instance using the configuration dictionary
config = generic.Config(config=config_data)

# Instantiate the Tokenizer using the configuration
tokenizer = Tokenizer(config)

# Extract numerical 'data' column from the dataframe
numerical_values = data['data'].values

# Function to round values to four significant figures
# Function to round values to four significant figures using progress bar
def round_to_4sf_with_progress(values):
    total = len(values)
    rounded_values = []

    # Iterate over each value and round to 4 significant figures
    for idx, value in enumerate(values):
        if value != 0:
            decimal_places = 3 - int(np.floor(np.log10(np.abs(value))))
            rounded_values.append(round(value, decimal_places))
        else:
            rounded_values.append(0)  # Keep zero as is
        generic.progress_bar(idx + 1, total, text="Rounding Values")

    return np.array(rounded_values)

# Round the numerical values to four significant figures with progress bar
rounded_values = round_to_4sf_with_progress(numerical_values)

# Tokenize the rounded values using progress bar
tokens_4sf = []
total_tokens = len(rounded_values)

for idx, value in enumerate(rounded_values):
    token = tokenizer.tokenize([value])[0]
    tokens_4sf.append(token)
    generic.progress_bar(idx + 1, total_tokens, text="Tokenizing Values")


# Add the new tokens to the dataframe
data['tokens_4sf'] = tokens_4sf

# Display the updated dataframe
print(data.head())

# If you'd like to save the result to a new CSV file
data.to_csv('amzn_2023_tokenized.csv', index=False)


