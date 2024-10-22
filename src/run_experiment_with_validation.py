# Assuming the code structure in src/
import sys
import os
import argparse
import pandas as pd

# Add src/ to the system path
sys.path.append('/workspace/IRP/src')

# Import local functions
from experiment_config import tic_list, result_dir, model_dir, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS
from training import train_baseline_agents, train_naive_strategy, train_ewc_agents, train_replay_agents
from data_processing import split_collect_stock_data_from_csv
from experiment_utils import update_combination_status
from envs import PortfolioAllocationEnv, PortfolioAllocationEnvLogReturn
from performance import test_agent_performance, calculate_performance_metrics

# Main Experiment Loop with incremental CSV writing
def experiment_iteration(model_dir, train_df1, train_df2, val_df1, val_df2, test_df1, test_df2, group1, group2, iteration, 
                         PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, results_file, returns_file,
                         validation_interval=10, patience=3, total_timesteps=[100000, 80000, 50000], 
                         env_class=PortfolioAllocationEnv):
    """
    Perform an experiment iteration by training PPO, A2C, and DDPG agents using Baseline, Naive, EWC, and Replay Buffer strategies.
    Results are written to the CSV file incrementally.

    Parameters:
    - model_dir: Directory to save/load models.
    - train_df1: DataFrame for training on group1.
    - train_df2: DataFrame for training on group2.
    - test_df1: DataFrame for testing on group1.
    - test_df2: DataFrame for testing on group2.
    - group1: List of tickers in group1.
    - group2: List of tickers in group2.
    - iteration: Current iteration number.
    - PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS: Model parameters for PPO, A2C, and DDPG.
    - results_file: Path to the CSV file for storing results incrementally.
    - returns_file: Path to the CSV file for result portfolio returns incrementally.

    Returns:
    - None (Results are written to the CSV file directly).
    """

    # Step 1: Baseline Training
    print(f"Iteration {iteration}: Training Baseline Agents")
    ppo_baseline, a2c_baseline, ddpg_baseline = train_baseline_agents(
        model_dir, train_df1, group1, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS,
        validation_df=val_df1, validation_interval=validation_interval, patience=patience, 
        total_timesteps=total_timesteps, env_class=env_class
    )

    # Step 2: Naive Strategy Training
    print(f"Iteration {iteration}: Training Naive Strategy Agents")
    ppo_naive, a2c_naive, ddpg_naive = train_naive_strategy(
        model_dir, train_df2, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS,
        validation_df=val_df2, validation_interval=validation_interval, patience=patience, 
        total_timesteps=total_timesteps, env_class=env_class
    )

    # Step 3: EWC Strategy Training
    print(f"Iteration {iteration}: Training EWC Strategy Agents")
    ppo_ewc, a2c_ewc, ddpg_ewc = train_ewc_agents(
        model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS,
        validation_df=val_df2, validation_interval=validation_interval, patience=patience, 
        total_timesteps=total_timesteps, env_class=env_class
    )

    # Step 4: Replay Buffer Strategy Training
    print(f"Iteration {iteration}: Training Replay Buffer Strategy Agents")
    ppo_replay, a2c_replay, ddpg_replay = train_replay_agents(
        model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS,
        validation_df=val_df2, validation_interval=validation_interval, patience=patience, 
        total_timesteps=total_timesteps, env_class=env_class
    ) 
   
    # Performance Testing
    print(f"Iteration {iteration}: Testing Performance of Trained Agents")
    test_returns = []
    test_results = []

    # Test each agent's performance on group1 and group2
    for strategy, ppo_agent, a2c_agent, ddpg_agent in [
        ("baseline", ppo_baseline, a2c_baseline, ddpg_baseline),
        ("naive", ppo_naive, a2c_naive, ddpg_naive),
        ("ewc", ppo_ewc, a2c_ewc, ddpg_ewc),
        ("replay", ppo_replay, a2c_replay, ddpg_replay)
    ]:
        env_group1 = PortfolioAllocationEnv(test_df1, initial_balance=100000, tic_list=group1, transaction_fee_rate=0.001)
        env_group2 = PortfolioAllocationEnv(test_df2, initial_balance=100000, tic_list=group1, transaction_fee_rate=0.001)

        for rl_model, agent in [
            ("PPO", ppo_agent),
            ("A2C", a2c_agent),
            ("DDPG", ddpg_agent)
        ]:
            
            # Setup the agent to run using Group1 test data
            agent.set_env(env_group1)
            group1_return, _ = test_agent_performance(agent=agent, env=env_group1)

            # Setup the agent to run using Group2 test data
            agent.set_env(env_group2)
            group2_return, _ = test_agent_performance(agent=agent, env=env_group2)

            # Calculate the portfolio performance metrics such as cummunclative retuens, Sharpe ratio, etc.
            metrics1 = calculate_performance_metrics(group1_return)
            metrics2 = calculate_performance_metrics(group2_return)

            # Add the run identification to the metrics
            metrics1['iteration'] = iteration
            metrics1['strategy'] = strategy
            metrics1['rl_model'] = rl_model
            metrics1['data_group'] = 'Group1'
            metrics2['iteration'] = iteration
            metrics2['strategy'] = strategy
            metrics2['rl_model'] = rl_model
            metrics2['data_group'] = 'Group2'

            # Add the preformance for Group1 and Group2 to results for analysis
            test_results.append(metrics1)
            test_results.append(metrics2)

            # Store the portfolio values series for future analysis
            test_returns.append({
                'iteration': iteration,
                'strategy': strategy,
                'rl_model': rl_model,
                'data_group': 'Group1',
                'portfolio_values': group1_return
            })
            test_returns.append({
                'iteration': iteration,
                'strategy': strategy,
                'rl_model': rl_model,
                'data_group': 'Group2',
                'portfolio_values': group2_return
            })



    # Convert results to a DataFrame for easier analysis
    results_df = pd.DataFrame(test_results)
    returns_df = pd.DataFrame(test_returns)

    # Append results to the CSV file
    if not os.path.isfile(results_file):
        results_df.to_csv(results_file, index=False, mode='w')  # Write header if file doesn't exist
    else:
        results_df.to_csv(results_file, index=False, mode='a', header=False)  # Append without writing header

    if not os.path.isfile(returns_file):
        returns_df.to_csv(returns_file, index=False, mode='w')  # Write header if file doesn't exist
    else:
        returns_df.to_csv(returns_file, index=False, mode='a', header=False)  # Append without writing header

def run_experiment_with_validation(combination_file='combinartions.csv'):
    """
    This is the main loop to run the experiment once.
    12 models (Baslein, Naive, EWC and Reply on PPO, A2C and DDPF) are trained
    24 tests (on Group 1 and Group 2 test data for 12 models) are done.
    """
    iteration, group1, group2, df1, df2 = split_collect_stock_data_from_csv(tic_list=tic_list, combination_file=combination_file)

    results_file = os.path.join(result_dir, f'results-viking-{iteration}.csv')
    returns_file = os.path.join(result_dir, f'returns-viking-{iteration}.csv')

    # Split the dataset based on the 'Date' or first level of the multi-index
    train_df1 = df1.loc[(df1.index.get_level_values(0) >= '2010-01-01') & (df1.index.get_level_values(0) <= '2019-12-31')]
    validation_df1 = df1.loc[(df1.index.get_level_values(0) >= '2020-01-01') & (df1.index.get_level_values(0) <= '2021-12-31')]
    trade_df1 = df1.loc[(df1.index.get_level_values(0) >= '2022-01-01') & (df1.index.get_level_values(0) <= '2023-12-31')]

    train_df2 = df2.loc[(df2.index.get_level_values(0) >= '2010-01-01') & (df2.index.get_level_values(0) <= '2019-12-31')]
    validation_df2 = df2.loc[(df2.index.get_level_values(0) >= '2020-01-01') & (df2.index.get_level_values(0) <= '2021-12-31')]
    trade_df2 = df2.loc[(df2.index.get_level_values(0) >= '2022-01-01') & (df2.index.get_level_values(0) <= '2023-12-31')]

    experiment_iteration(
        model_dir, train_df1, train_df2, trade_df1, validation_df1, validation_df2, trade_df2, group1, group2, 
        iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, results_file, returns_file,
        validation_interval=20, patience=3, total_timesteps=[50000, 80000, 50000], 
        env_class=PortfolioAllocationEnvLogReturn
    )

    update_combination_status(iteration, "completed", csv_file=combination_file)

    return iteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment with optional parameters.")
    parser.add_argument('--combination_file', type=str, default='combinations.csv', help="Path to the combination file.")
    args = parser.parse_args()

    run_experiment_with_validation(combination_file=args.combination_file)
