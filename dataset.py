import pandas as pd
import numpy as np
import yfinance as yf

def RDP(dataframe, lags):
    """
    Compute the Relative Difference in Percentage (RDP) lagged for specified periods.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing at least a 'Close' column.
    - lags (list): A list of integers representing the lag periods for RDP computation.

    Returns:
    - pd.DataFrame: The input DataFrame with new columns for each computed RDP.
    """
    if 'Close' not in dataframe.columns:
        raise ValueError("The input DataFrame must contain a 'Close' column.")

    for i in range(len(lags)):
        lag = lags[i] 
        column_name = f'RDP_{lag}'
        dataframe[column_name] = (dataframe['Close'].shift(lag - (i+1)*5) - dataframe['Close'].shift(lag)) / dataframe['Close'].shift(lag) * 100

    return dataframe

def process_sp500_data(start_date, end_date, lags, ticker='^GSPC'):
    """
    Process S&P 500 data by downloading it, calculating returns, adding lagged features,
    calculating rolling volatility, calculating SMAs, flattening columns, and cleaning up the dataset.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., '^GSPC' for S&P 500).
    - start_date (str): Start date for the data download (YYYY-MM-DD).
    - end_date (str): End date for the data download (YYYY-MM-DD).
    - lags (list): Number of lagged features to create.

    Returns:
    - pd.DataFrame: Processed DataFrame with cleaned and lagged features.
    """
    # Download the data
    sp_500 = yf.download(ticker, start=start_date, end=end_date)
    
    # Add the RDP columns
    sp_500 = RDP(sp_500, lags)

    # Compute EMA3 and RDP+5
    sp_500['EMA3'] = sp_500['Close'].ewm(span=3, adjust=False).mean()
    sp_500['RDP+5'] = (sp_500['EMA3'].shift(-5) - sp_500['EMA3']) / sp_500['EMA3'] * 100

    # Compute EMA15
    sp_500['EMA15'] = sp_500['Close'] - sp_500['Close'].ewm(span=15, adjust=False).mean()

    # Select relevant columns
    sp_500 = sp_500[['Close', 'EMA15'] + [f'RDP_{lag}' for lag in lags] + ['RDP+5']]

    # Flatten MultiIndex columns if they exist
    if isinstance(sp_500.columns, pd.MultiIndex):
        sp_500.columns = ['_'.join(filter(None, col)) for col in sp_500.columns]

    # Remove columns containing "Ticker"
    sp_500 = sp_500.loc[:, ~sp_500.columns.str.contains("Ticker")]

    # Rename Volume column if needed
    sp_500.rename(columns={"Close_^GSPC": "Close"}, inplace=True)

    # Drop rows with NaN values
    sp_500.dropna(inplace=True)

    return sp_500

# Example usage
if __name__ == "__main__":

    # Parameters
    start_date = "2005-01-01"
    end_date = "2024-01-01"
    lags = [5, 10, 15, 20]

    # Process S&P 500 data
    sp500_data = process_sp500_data(start_date=start_date, end_date=end_date, lags=lags)
    print("S&P 500 data processed with RDP.")

    # Save the processed data
    sp500_data.to_csv("data/processed_data_rdp.csv", index=True)
    print("Data saved as 'processed_data_rdp.csv'.")
