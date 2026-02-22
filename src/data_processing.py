import yfinance as yf
import pandas as pd
import talib
import numpy as np
import random
import os

#Local Import
from experiment_config import data_dir
from experiment_utils import get_next_combination

def _calculate_vortex(high, low, close, length=14):
    """
    Compute Vortex Indicator (VI+ and VI-) using a pandas_ta-compatible formula.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(length, min_periods=length).sum()
    vi_plus = vm_plus.rolling(length, min_periods=length).sum() / tr_sum
    vi_minus = vm_minus.rolling(length, min_periods=length).sum() / tr_sum

    return vi_plus, vi_minus


def get_data_from_yahoo(tic_list, start_date='2009-01-01', end_date='2024-01-01'):
    """
    Fetch OHLCV data from Yahoo Finance for a list of tickers and fill missing data.

    Args:
    - tic_list (list): List of stock tickers.
    - start_date (str): Start date for fetching data.
    - end_date (str): End date for fetching data.

    Returns:
    - pd.DataFrame: DataFrame containing OHLCV data for all tickers with missing data filled.
    """
    # Download historical data for all stocks
    all_df = pd.DataFrame()
    for tic in tic_list:
        df = yf.download(tic, start=start_date, end=end_date)
        df['tic'] = tic
        df = df.ffill()
        all_df = pd.concat([all_df, df])

    all_df = all_df.reset_index()
    all_df = all_df.set_index(['tic'])
    all_df = all_df.sort_index()  # Sort by date and ticker

    return all_df


def add_technical_indicators(df):
    """
    Add common technical indicators and calculate the covariance matrix for each stock pair.

    Args:
    - df (pd.DataFrame): DataFrame with OHLCV data.

    Returns:
    - pd.DataFrame: DataFrame with technical indicators and covariance matrix added.
    """
    # Work in flat form for robust per-ticker assignment, then restore index='tic'
    df = df.reset_index()
    df = df.sort_values(by=['tic', 'Date']).copy()

    indicator_cols = [
        'SMA_10', 'SMA_50', 'SMA_200', 'EMA_10', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_hist',
        'BBL', 'BBM', 'BBU',
        'ATR_14', 'Stochastic_K', 'Stochastic_D',
        'CCI_14', 'ADX_14', 'OBV', 'WILLR_14', 'ROC_10',
        'Vortex_Pos', 'Vortex_Neg',
    ]
    for col in indicator_cols:
        df[col] = np.nan

    for _, idx in df.groupby('tic').groups.items():
        g = df.loc[idx].sort_values('Date')

        close = g['Close'].astype(float)
        high = g['High'].astype(float)
        low = g['Low'].astype(float)
        volume = g['Volume'].astype(float)

        df.loc[g.index, 'SMA_10'] = talib.SMA(close.values, timeperiod=10)
        df.loc[g.index, 'SMA_50'] = talib.SMA(close.values, timeperiod=50)
        df.loc[g.index, 'SMA_200'] = talib.SMA(close.values, timeperiod=200)
        df.loc[g.index, 'EMA_10'] = talib.EMA(close.values, timeperiod=10)
        df.loc[g.index, 'RSI_14'] = talib.RSI(close.values, timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        df.loc[g.index, 'MACD'] = macd
        df.loc[g.index, 'MACD_signal'] = macd_signal
        df.loc[g.index, 'MACD_hist'] = macd_hist

        bbu, bbm, bbl = talib.BBANDS(
            close.values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0,
        )
        df.loc[g.index, 'BBL'] = bbl
        df.loc[g.index, 'BBM'] = bbm
        df.loc[g.index, 'BBU'] = bbu

        df.loc[g.index, 'ATR_14'] = talib.ATR(high.values, low.values, close.values, timeperiod=14)

        stoch_k, stoch_d = talib.STOCH(
            high.values,
            low.values,
            close.values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        df.loc[g.index, 'Stochastic_K'] = stoch_k
        df.loc[g.index, 'Stochastic_D'] = stoch_d

        df.loc[g.index, 'CCI_14'] = talib.CCI(high.values, low.values, close.values, timeperiod=14)
        df.loc[g.index, 'ADX_14'] = talib.ADX(high.values, low.values, close.values, timeperiod=14)
        df.loc[g.index, 'OBV'] = talib.OBV(close.values, volume.values)
        df.loc[g.index, 'WILLR_14'] = talib.WILLR(high.values, low.values, close.values, timeperiod=14)
        df.loc[g.index, 'ROC_10'] = talib.ROC(close.values, timeperiod=10)

        vi_pos, vi_neg = _calculate_vortex(high, low, close, length=14)
        df.loc[g.index, 'Vortex_Pos'] = vi_pos.values
        df.loc[g.index, 'Vortex_Neg'] = vi_neg.values

    df = df.set_index('tic')
    return df

def add_covariance_matrix(df, tic_list, window=252):
    """
    Calculate the covariance matrix for each stock pair in the given time window.

    Args:
    - df (pd.DataFrame): DataFrame with OHLCV data.
    - tic_list (list): List of stock tickers.
    - window (int): Time window for calculating covariance matrix. default is 252 trading days in a year

    Returns:
    - pd.DataFrame: DataFrame with covariances are added per stock per date
    """

    # Calculate the covariance matrix for the past year for each stock pair
    # We'll use a rolling window of 252 trading days (1 year)
    # window = 252  # 252 trading days in a year
    cov_df = pd.DataFrame()

    # Pivot the data to create a wide-format DataFramce for calculating covariance
    df = df.reset_index()
    Close_prices = df.pivot(index='Date',columns='tic', values='Close')

    for i in range(window, len(Close_prices)):
      cov_matrix = Close_prices.iloc[i-window:i].pct_change().dropna()
      cov_matrix = cov_matrix.cov()
      cov_matrix['Date'] = Close_prices.index[i]
      cov_df = pd.concat([cov_df, cov_matrix])

    cov_df = cov_df.reset_index()
    cov_df = cov_df.set_index(['Date', 'tic'])
    df = df.set_index(['Date', 'tic'])
    # Add the covariance matrix to the DataFrame
    df = pd.concat([df, cov_df], axis=1)

    # Fill any remaining NaNs after adding indicators
    df = df.fillna(0)

    return df

def split_collect_stock_data(tic_list, start_date='2009-01-01', end_date='2024-01-01'):
    """
    The function is to randomly generate 2 sets of stock trading data.
    
    Parameters:
    tic_list : A list of all stocks for collecting trading data
    start_date : The start date of the trading data
    end_date: The end date of the trading data

    Returns:
    group1, group2: the tickers list of the 2 data sets.
    df1, df2: 2 dataframes contain the trading data, technical indicators, and covariance.
    """

    tic_list_copy = tic_list.copy()
    random.shuffle(tic_list_copy)
    index = int(len(tic_list_copy)/2)
    group1 = tic_list_copy[:index]
    group2 = tic_list_copy[index:2*index]

    # sort group1 and group2
    group1.sort()
    group2.sort()

    df1 = get_data_from_yahoo(group1, start_date, end_date)
    df1 = add_technical_indicators(df1)
    df1 = add_covariance_matrix(df1, group1)

    df2 = get_data_from_yahoo(group2, start_date, end_date)
    df2 = add_technical_indicators(df2)
    df2 = add_covariance_matrix(df2, group2)

    return group1, group2, df1, df2

def split_collect_stock_data_from_csv(tic_list, csv_file='dji_stock_data.csv', combination_file='combinations.csv', start_date='2009-01-01', end_date='2024-01-01'):
    """
    Split the pre-collected stock data into group1 and group2 using pre-calculated combinations from a CSV file.
    
    Parameters:
    - tic_list: A list of all stocks (used for validation, not for splitting).
    - csv_file: The CSV file containing pre-collected stock data.
    - combination_file: The CSV file containing the pre-calculated combinations.
    - start_date: The start date of the data to filter the data from the CSV.
    - end_date: The end date of the data to filter the data from the CSV.

    Returns:
    - iteration: The iteration number
    - group1, group2: The tickers for group1 and group2.
    - df1, df2: DataFrames for group1 and group2 with technical indicators and covariance.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Load the stock data from the CSV if it exists
    if os.path.exists(os.path.join(data_dir, csv_file)):
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        print(f"Loaded stock data from {csv_file}")
    else:
        df = get_data_from_yahoo(tic_list, start_date=start_date, end_date=end_date)
        df = df.reset_index()
        print(f"Collect data from Yahoo")
        df.to_csv(os.path.join(data_dir, csv_file), index=False)

    # Filter the data by date range
    df = df.set_index('tic')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Get the next untrained combination of group1 and group2
    iteration, group1, group2 = get_next_combination(tic_list, combination_file)

    # Create df1 and df2 based on the 'tic' index (ticker symbol)
    df = df.reset_index()
    df1 = df[df['tic'].isin(group1)].copy()
    df2 = df[df['tic'].isin(group2)].copy()

    df1 = df1.reset_index()
    df1 = df1.set_index('tic')

    df2 = df2.reset_index()
    df2 = df2.set_index('tic')

    # Add technical indicators and covariance to df1 and df2
    df1 = add_technical_indicators(df1)
    df1 = add_covariance_matrix(df1, group1)

    df2 = add_technical_indicators(df2)
    df2 = add_covariance_matrix(df2, group2)

    return iteration, group1, group2, df1, df2
