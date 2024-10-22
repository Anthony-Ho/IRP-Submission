PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}

A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.005, 
    "learning_rate": 0.0002
}

DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.001
}

tic_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'JNJ', 'V',
            'WMT', 'JPM', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'BAC', 'VZ', 'KO',
            'ADBE', 'NFLX', 'PFE', 'INTC', 'CSCO', 'PEP', 'MRK', 'ABT', 'T', 'XOM']


TRANSACTION_FEE_RATE = 0.001
INITIAL_BALANCE = 100000

data_dir = '../data'
model_dir = '../models'
result_dir = '../results'
