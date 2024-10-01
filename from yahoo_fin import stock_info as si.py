from yahoo_fin import stock_info as si
import pandas as pd

# Get the list of tickers for a specific exchange (e.g., NASDAQ)
nasdaq_tickers = si.tickers_nasdaq(True)

# Initialize lists to store data
stock_codes = []
stock_names = []
stock_exchanges = []

print(nasdaq_tickers[:10])
'''
# Loop through the tickers and get the company name and exchange
for ticker in nasdaq_tickers:  # Fetching the first 100 items as an example
    try:
        info = si.get_company_info(ticker)
        stock_codes.append(ticker)
        stock_names.append(info.loc['longBusinessSummary'].values[0] if 'longBusinessSummary' in info.index else 'N/A')
        stock_exchanges.append('NASDAQ')
    except Exception as e:
        # Handle exceptions (e.g., ticker not found, etc.)
        print(f"Error retrieving data for {ticker}: {e}")

# Create a DataFrame from the data
df = pd.DataFrame({
    'Stock Code': stock_codes,
    'Stock Name': stock_names,
    'Stock Exchange': stock_exchanges
})

# Display or save the DataFrame
print(df)
# Optionally, save to a CSV file
# df.to_csv("nasdaq_stock_data.csv", index=False)
'''