import pandas as pd
import yfinance as yf
from datasets import download_fred_data

def monthly_to_daily(dataframe):
    """
    Convert monthly data to daily data by forward-filling values.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with datetime index (monthly data).

    Returns:
    - pd.DataFrame: DataFrame with daily data.
    """
    # Ensure the date column is in datetime format
    dataframe.index = pd.to_datetime(dataframe.index)

    # Create daily date range covering the entire range of the input data
    daily_index = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq='B')

    # Reindex to daily frequency and forward-fill the values
    daily_data = dataframe.reindex(daily_index, method='ffill')
    daily_data.index.name = "Date"

    return daily_data

def create_lagged_features(dataframe, value_column, n_lags):
    """
    Create lagged features for a given value column in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with datetime index and the target value column.
    - value_column (str): Column name of the values for which lagged features are created.
    - n_lags (int): Number of lagged features to generate.

    Returns:
    - pd.DataFrame: DataFrame with lagged features added as new columns.
    """
    for lag in range(1, n_lags + 1):
        dataframe[f"Lag_{lag}"] = dataframe[value_column].shift(lag)
    dataframe.dropna(inplace=True)  # Drop rows with NaN values caused by shifting
    return dataframe

def calculate_rsi(data, column='returns', window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given data column.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - column (str): Column for which RSI is calculated.
    - window (int): Window size for RSI calculation.

    Returns:
    - pd.Series: RSI values.
    """
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def process_sp500_data(start_date, end_date, n_lags, vol_lag, SMA_lags, rsi_window,ticker='^GSPC'):
    """
    Process S&P 500 data by downloading it, calculating returns, adding lagged features,
    calculating rolling volatility, calculating SMAs, flattening columns, and cleaning up the dataset.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., '^GSPC' for S&P 500).
    - start_date (str): Start date for the data download (YYYY-MM-DD).
    - end_date (str): End date for the data download (YYYY-MM-DD).
    - n_lags (int): Number of lagged features to create.
    - vol_lag (int): Rolling window size for calculating volatility.
    - SMA_lags (int or list): Single value or list of window sizes for calculating SMAs.
    - rsi_window (int): Rolling window size for calculating rsi.

    Returns:
    - pd.DataFrame: Processed DataFrame with cleaned and lagged features.
    """
    # Download the data
    sp_500 = yf.download(ticker, start=start_date, end=end_date)

    # Calculate daily returns
    sp_500['returns'] = sp_500['Adj Close'].pct_change()

    # Calculate rolling volatility
    sp_500[f'volatility_lag_{vol_lag}'] = sp_500['returns'].rolling(window=vol_lag).std()
    
    # Calculate RSI
    sp_500[f'RSI_{rsi_window}'] = calculate_rsi(sp_500, column='returns', window=rsi_window)

    # Calculate SMAs
    if isinstance(SMA_lags, int):
        SMA_lags = [SMA_lags]
    for lag in SMA_lags:
        sp_500[f'SMA_{lag}'] = sp_500['returns'].rolling(window=lag).mean()

    # Select relevant columns
    sp_500 = sp_500[['returns', 'Volume', f'volatility_lag_{vol_lag}'] + [f'SMA_{lag}' for lag in SMA_lags] + [f'RSI_{rsi_window}']]

    # Create lagged features
    sp_500 = create_lagged_features(dataframe=sp_500, value_column='returns', n_lags=n_lags)

    # Flatten MultiIndex columns if they exist
    if isinstance(sp_500.columns, pd.MultiIndex):
        sp_500.columns = ['_'.join(filter(None, col)) for col in sp_500.columns]

    # Remove columns containing "Ticker"
    sp_500 = sp_500.loc[:, ~sp_500.columns.str.contains("Ticker")]

    # Rename Volume column if needed
    sp_500.rename(columns={"Volume_^GSPC": "volume"}, inplace=True)

    # Drop rows with NaN values
    sp_500.dropna(inplace=True)

    return sp_500

def load_macro_data(start_date, end_date):
    """
    Load macroeconomic data for unemployment rate (UNRATE) and CPI.

    Parameters:
    - start_date (str): Start date for the data download (YYYY-MM-DD).
    - end_date (str): End date for the data download (YYYY-MM-DD).

    Returns:
    - pd.DataFrame: Combined DataFrame with daily macroeconomic data.
    """
    # Load UNRATE data
    df_unrate = download_fred_data(
        start_date=start_date,
        end_date=end_date,
        filename="data/fred_unrate.csv",
        indicator_id="UNRATE",
        overwrite=True,
    )
    df_unrate = monthly_to_daily(df_unrate)

    ''' 
    The Sticky Price Consumer Price Index (CPI) is calculated from a subset of goods and services included in the CPI that change price relatively infrequently. 
    Because these goods and services change price relatively infrequently, they are thought to incorporate expectations about future inflation to a greater 
    degree than prices that change on a more frequent basis.
    One possible explanation for sticky prices could be the costs firms incur when changing price. 
    '''
    # Load CPI data
    df_cpi = download_fred_data(
        start_date=start_date,
        end_date=end_date,
        filename="data/fred_cpi.csv",
        indicator_id="CORESTICKM159SFRBATL",
        overwrite=True,
    )
    df_cpi = monthly_to_daily(df_cpi)
    df_cpi.rename(columns={"CORESTICKM159SFRBATL": "CPI"}, inplace=True)

    # Merge dataframes
    combined_df = pd.merge(df_unrate, df_cpi, left_index=True, right_index=True, how="outer")

    return combined_df

def merge_dataframes(sp500_data, macro_data):
    """
    Merge the output DataFrames of process_sp500_data and load_macro_data on the 'Date' index.

    Parameters:
    - sp500_data (pd.DataFrame): DataFrame containing processed S&P 500 data.
    - macro_data (pd.DataFrame): DataFrame containing macroeconomic data.

    Returns:
    - pd.DataFrame: Merged DataFrame containing both S&P 500 and macroeconomic data.
    """
    sp500_data.index.name = "Date"
    macro_data.index.name = "Date"
    merged_df = pd.merge(sp500_data, macro_data, left_index=True, right_index=True, how="outer")
    merged_df.dropna(inplace=True)  # Drop rows with NaN values after merging
    return merged_df

# Example usage
if __name__ == "__main__":
    # Parameters
    start_date = "2010-01-01"
    end_date = "2024-01-01"
    n_lags = 3
    vol_lag = 10
    SMA_lags = [10, 20]
    rsi_window = 14

    # Load macro data
    macro_data = load_macro_data(start_date=start_date, end_date=end_date)
    print("Macro data loaded.")

    # Process S&P 500 data
    sp500_data = process_sp500_data(start_date=start_date, end_date=end_date, n_lags=n_lags, vol_lag=vol_lag, SMA_lags=SMA_lags, rsi_window=rsi_window)
    print("S&P 500 data processed.")

    # Merge the dataframes
    merged_data = merge_dataframes(sp500_data, macro_data)

    # Save the merged data
    merged_data.to_csv("data/merged_data.csv", index=True)
    print("Merged data saved as 'merged_data.csv'.")