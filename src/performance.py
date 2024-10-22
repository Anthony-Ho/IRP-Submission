import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def validate_agent_performance(agent, env, episodes=1, is_vec_env=False):
    """
    Test the agent on the given environment for a number of episodes.
    Return the average cumulative return or other metrics.
    If the environment is a VecEnv, set is_vec_env=True.
    """
    all_rewards = []

    for _ in range(episodes):
        if is_vec_env:
            obs = env.reset()  # VecEnv reset (no tuple)
        else:
            obs, _ = env.reset()  # Gym API reset

        done = False
        total_reward = 0

        while not done:
            action, _ = agent.predict(obs)
            if is_vec_env:
                obs, reward, done, info = env.step(action)  # VecEnv step (no tuple)
                done = done[0]  # For VecEnv, done is a list/array
                reward = reward[0]  # For VecEnv, reward is a list/array
            else:
                obs, reward, done, _, _ = env.step(action)  # Gym API step

            total_reward += reward

        all_rewards.append(total_reward)
    
    # Return the average performance metric (e.g., cumulative return)
    return np.mean(all_rewards)

def test_agent_performance(agent, env):
    """
    Test the agent on the test dataset and track portfolio values indexed by date.
    """
    obs, _ = env.reset()
    done = False

    # Initialize a dictionary to store portfolio values indexed by date
    portfolio_values = {}
    
    # Get the list of dates from the test environment's dataframe (index level 0)
    dates = env.df.index.get_level_values(0).unique()

    # Add the initial portfolio value with the first date
    portfolio_values[dates[0]] = env.initial_balance

    step = 0
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        step += 1

        # Record the portfolio value for the current date
        portfolio_values[dates[step]] = env.portfolio_value

    # Convert the dictionary to a Pandas Series for easy plotting and comparison
    portfolio_values_series = pd.Series(portfolio_values)

    # Calculate cumulative return
    cumulative_return = (portfolio_values_series.iloc[-1] / portfolio_values_series.iloc[0]) - 1

    return portfolio_values_series, cumulative_return

def plot_agent_performance_multiple_envs(model_files, envs, benchmarks, title):
    """
    General function to plot cumulative returns for multiple agents over multiple environments,
    handling MultiIndex DataFrames with 'date' in the index, and displaying dates on the x-axis.
    
    Parameters:
    - model_files: dict, {name: (model_class, filename)}, where name is a string identifier for the agent,
      model_class is the class (e.g., PPO, A2C, DDPG), and filename is the path to the saved model.
    - envs: dict, {env_name: environment}, where env_name is a string identifier for the environment.
    - benchmarks: dict, {env_name: (benchmark_name, benchmark_prices)}, where benchmark_name is the benchmark name,
      benchmark_prices is a pandas Series with datetime index of benchmark prices.
    - title: string, the title of the plot.
    """

    num_envs = len(envs)
    fig, axes = plt.subplots(1, num_envs, figsize=(12 * num_envs, 10), sharex=False)
    if num_envs == 1:
        axes = [axes]

    for idx, (env_name, env) in enumerate(envs.items()):
        agent_cumulative_returns = {}
        for name, (model_class, filename) in model_files.items():
            # Load the model
            model = model_class.load(filename, env=env)

            # Test the agent performance
            portfolio_values, _ = test_agent_performance(model, env)

            # Calculate cumulative returns over time
            cumulative_returns = (portfolio_values / portfolio_values.iloc[0]) - 1

            # Since cumulative_returns is already indexed by dates, we can use it directly
            agent_cumulative_returns[name] = cumulative_returns

        # Get benchmark for this environment
        benchmark_name, benchmark_prices = benchmarks[env_name]

        # Align benchmark cumulative returns with the test period
        env_dates = agent_cumulative_returns[next(iter(agent_cumulative_returns))].index
        benchmark_period = benchmark_prices.loc[benchmark_prices.index.isin(env_dates)]

        # Handle potential missing dates by reindexing
        benchmark_period = benchmark_period.reindex(env_dates, method='ffill')

        # Calculate cumulative returns
        benchmark_cumulative_returns = (benchmark_period / benchmark_period.iloc[0]) - 1

        # Plot cumulative returns
        ax = axes[idx]
        for name, cumulative_returns in agent_cumulative_returns.items():
            ax.plot(cumulative_returns.index, cumulative_returns.values, label=name)

        # Add benchmark cumulative returns
        ax.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns.values,
                label=benchmark_name, linestyle="--")

        ax.set_title(f"{title} - {env_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True)

        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(portfolio_values, benchmark_values=None, risk_free_rate=0.0):
    """
    Calculate portfolio performance metrics based on time series of portfolio values.
    
    Parameters:
    - portfolio_values: pandas Series of portfolio values over time.
    - benchmark_values: (optional) pandas Series of benchmark values (e.g., index like DJI) over time.
    - risk_free_rate: (optional) Risk-free rate (annualized). Default is 0.0.
    
    Returns:
    - metrics: A dictionary containing performance metrics.
    """
    # Ensure portfolio values are a pandas Series
    portfolio_values = pd.Series(portfolio_values)
    
    # Calculate daily returns
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    # Cumulative return
    cumulative_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

    # Average daily return
    avg_daily_return = portfolio_returns.mean()
    
    # Annualized return
    annualized_return = (1 + avg_daily_return) ** 252 - 1
    
    # Volatility (annualized standard deviation of daily returns)
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else np.nan
    
    # Sortino Ratio (using downside risk)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
    
    # Maximum Drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # Alpha and Beta (if benchmark is provided)
    alpha, beta, treynor_ratio = np.nan, np.nan, np.nan
    if benchmark_values is not None:
        benchmark_returns = pd.Series(benchmark_values).pct_change().dropna()
        cov_matrix = np.cov(portfolio_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / np.var(benchmark_returns)
        alpha = (annualized_return - risk_free_rate) - (beta * (np.mean(benchmark_returns) * 252 - risk_free_rate))
        treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else np.nan

    # Value at Risk (VaR) at 95% confidence level using historical method
    var_95 = np.percentile(portfolio_returns, 5)
    
    # Compile metrics into a dictionary
    metrics = {
        'Cumulative Return': cumulative_return,
        'Volatility (Standard Deviation of Returns)': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Alpha': alpha,
        'Beta': beta,
        'VaR (95%)': var_95
    }
    
    return metrics