import pandas as pd
import os
import sys

# Add src/ to the system path
sys.path.append('src')

from data_processing import split_collect_stock_data_from_csv
from experiment_config import tic_list

# Create a dummy combinations.csv if it doesn't exist
if not os.path.exists('data/combinations-test.csv'):
    os.makedirs('data', exist_ok=True)
    with open('data/combinations-test.csv', 'w') as f:
        f.write("iteration;group1;group2;status\n")
        f.write("1;['AAPL', 'MSFT'];['GOOGL', 'AMZN'];untrained\n")

# Make sure status is untrained to ensure lock acquisition
try:
    df = pd.read_csv('data/combinations-test.csv', sep=';')
except:
    df = pd.read_csv('data/combinations-test.csv')

if df.iloc[0]['status'] != 'untrained':
    df.loc[0, 'status'] = 'untrained'
    df.to_csv('data/combinations-test.csv', index=False, sep=';')
    # clean up lock file if exists
    if os.path.exists('data/lockfile_1.lck'):
        os.remove('data/lockfile_1.lck')


try:
    # Create dummy dji_stock_data.csv
    dates = pd.date_range(start='2010-01-01', periods=100, freq='D')
    dummy_data = []
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for tic in tickers:
        for date in dates:
            dummy_data.append({
                'Date': date,
                'tic': tic,
                'Open': 100, 'High': 110, 'Low': 90, 'Close': 105, 'Volume': 1000
            })

    pd.DataFrame(dummy_data).to_csv('data/dji_stock_data.csv', index=False)

    iteration, group1, group2, df1, df2, df_combined = split_collect_stock_data_from_csv(
        tic_list=tic_list,
        csv_file='dji_stock_data.csv',
        combination_file='combinations-test.csv',
        start_date='2010-01-01',
        end_date='2010-04-01' # short range
    )

    print("Successfully unpacked 6 values.")
    print(f"Group1: {len(group1)}")
    print(f"Group2: {len(group2)}")
    print(f"DF Combined Shape: {df_combined.shape}")
    print(f"DF Combined Index Names: {df_combined.index.names}")


except Exception as e:
    import traceback
    traceback.print_exc()
