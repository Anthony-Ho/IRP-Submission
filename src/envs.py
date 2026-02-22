import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioAllocationEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a stock portfolio allocation environment.
    """
    def __init__(
        self,
        df,
        initial_balance=10000,
        tic_list=None,
        transaction_fee_rate=0.001,
        logfile=None,
        verbose=False,
        print_every_n_episodes=5,
        random_start=False,
        random_window_size=False,
        min_window_size=200,
        max_window_size=None,
        window_size=None,
    ):
        """
        Initializes the environment.
        df = the stock trading information and features for training the agents.
        initial_balance = the starting balance for creating the portfolio.
        tic_list = the list of stocks to be included in the portfolio.
        """
        super(PortfolioAllocationEnv, self).__init__()

        # Stock Data
        self.df = df
        self.tic_list = tic_list
        self.n_stocks = len(self.tic_list)

        #Initial Setup
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.done = False
        self.episode = 0

        # Porfolio Weights (initially all cash)
        self.portfolio_weights = np.array([1] + [0] * self.n_stocks, dtype=float)
        self.holdings = np.zeros(self.n_stocks, dtype=float)
        # current_date = self.df.index.get_level_values(0).unique()[self.current_step]
        # current_data = self.df.loc[current_date]
        # self.prices = current_data['Close'].values
        self.portfolio_value = self.balance
        self.trading_dates = self.df.index.get_level_values(0).unique()
        self.episode_start_step = 0
        self.episode_end_step = len(self.trading_dates) - 1

        # Episode window controls
        self.random_start = random_start
        self.random_window_size = random_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.window_size = window_size

        # Transaction fee rate
        self.transaction_fee_rate = transaction_fee_rate

        # Action Space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks + 1,), dtype=np.float64)


        # Observation Space
        # All stocks' technical indicators and covariance matrix are embeded in df.
        # Therefore, the features for a trading day is n_stock * len(df.columns).
        # Amount Cash balance is also included.
        # Portfolio weight is also included, n_stocks + 1, where 1 is the case balance.
        self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.n_stocks * len(df.columns) + 1 + self.n_stocks + 1,),
        dtype=np.float64
        )

        # Memory for traking
        self._action_memory = []
        self._portfolio_value_memory = [self.balance]
        self._reward_memory = []
        self._final_weights_memory = [self.portfolio_weights]
        self._transaction_fees = []
        self.logfile = logfile
        self.verbose = verbose
        self.print_every_n_episodes = print_every_n_episodes

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        # Seed the environment (ensure reproducibility)
        if seed is not None:
            np.random.seed(seed)

        if (
            self.verbose
            and self.print_every_n_episodes > 0
            and (self.episode % self.print_every_n_episodes) == 0
            and (self.episode > 0)
        ):
            performance_metric = self._calculate_portfolio_metrics()
            print(performance_metric)
            # Keep logging to CSV if enabled, but avoid noisy DataFrame prints.
            self.log_episode()

        total_days = len(self.trading_dates)
        if total_days < 2:
            raise ValueError("At least 2 trading days are required for t+1 execution.")
        if (self.random_window_size or self.window_size is not None) and total_days < self.min_window_size:
            raise ValueError(
                f"Insufficient trading days ({total_days}) for minimum window size {self.min_window_size}."
            )

        # Resolve episode window size (number of dates included in the episode).
        effective_max_window = total_days if self.max_window_size is None else min(self.max_window_size, total_days)
        effective_min_window = min(max(2, self.min_window_size), effective_max_window)

        if self.random_window_size:
            selected_window = np.random.randint(effective_min_window, effective_max_window + 1)
        elif self.window_size is not None:
            selected_window = int(np.clip(self.window_size, effective_min_window, effective_max_window))
        else:
            selected_window = total_days

        max_start_step = total_days - selected_window
        if self.random_start and max_start_step > 0:
            self.episode_start_step = np.random.randint(0, max_start_step + 1)
        else:
            self.episode_start_step = 0
        self.episode_end_step = self.episode_start_step + selected_window - 1

        # reset environment state
        self.balance = self.initial_balance
        self.current_step = self.episode_start_step
        self.done = False
        self.portfolio_weights = np.array([1] + [0] * self.n_stocks, dtype=float)
        self.holdings = np.zeros(self.n_stocks, dtype=float)
        self.portfolio_value = self.balance
        self._action_memory = []
        self._portfolio_value_memory = [self.balance]
        self._reward_memory = []
        self._final_weights_memory = [self.portfolio_weights]
        self._transaction_fees = []
        self.episode += 1

        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
        - action: The portfolio weights (cash + stocks) provided by the agent.

        Returns:
        - observation: The new state after the action.
        - reward: The reward for the current step.
        - done: Whether the episode is over.
        - info: Additional information (empty in this case).
        """
        # Decide at day t, execute once at day t+1 open, then mark-to-market
        # at day t+1 close.
        previous_portfolio_value = self.portfolio_value

        if self.current_step >= self.episode_end_step:
            observation = self._get_observation()
            return observation, 0.0, True, False, {}

        execution_step = self.current_step + 1
        execution_open_prices = self._get_open_prices(execution_step)

        # First, apply overnight move to value holdings at the execution open.
        self._update_portfolio_value(execution_open_prices)

        # Then rebalance at that open using the up-to-date portfolio value.
        self._take_action(action, execution_open_prices)

        # Move to execution day and value the portfolio at that day's close.
        self.current_step = execution_step
        close_prices = self._get_close_prices(self.current_step)
        self._update_portfolio_value(close_prices)

        # Calculate reward based on the updated portfolio value
        reward = self._calculate_reward(previous_portfolio_value=previous_portfolio_value)

        # Store the action in memory
        self._action_memory.append(action)
        self._reward_memory.append(reward)  # Store the reward in memory

        # Store the updated portfolio value in memory
        self._portfolio_value_memory.append(self.portfolio_value)

        # Store the updated portfolio weights in memory
        self._final_weights_memory.append(self.portfolio_weights.copy())

        # Check if we're
        terminated = self.current_step >= self.episode_end_step
        truncated = False  # This could be set to True if truncate the episode would be used

        # Get the new observation
        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def _normalize_action(self, action):
        """
        Normalize the action (weights) to ensure they sum to 1.
        The action represents the target portfolio weights for cash and stocks.

        Args:
        - action: A raw action array of shape (n_stocks + 1,), where the first element
                    is the cash allocation and the remaining elements are for the stocks.

        Returns:
        - A normalized action (portfolio weights) that sum to 1.
        """
        # Clip the action to ensure no negative weights
        action = np.clip(action, 1e-8, None)

        # Ensure sufficient cash balance at step 0 to pay the transaction fee when initial
        # position setup.
        if (self.current_step == 0):
            if ((action[0] / np.sum(action)) < 0.01): # less than 1% cash
                # print(f"Too little cash allocated: {action[0]} / {np.sum(action)}.")
                action[0] = np.sum(action[1:]) / 99 # make cash to be 1%
            
        # Normalize the action so that the sum of the weights equals 1
        # Using softmax function that ensures all action elements are > 0 and the sum equals 1
        exp_action = np.exp(action - np.max(action))  # Subtract max to avoid overflow
        return exp_action / np.sum(exp_action)

    def _get_open_prices(self, step):
        """
        Get the 'Open' prices of all tickers at the specified step (date).
        """
        # Get the current date corresponding to the step
        current_date = self.df.index.get_level_values(0).unique()[step]

        # Filter the DataFrame for this date (all tickers)
        current_data = self.df.loc[current_date]

        # Extract the 'Open' prices for all tickers on the current date
        open_prices = current_data['Open']

        return open_prices

    def _get_close_prices(self, step):
        """
        Get the 'Close' prices of all tickers at the specified step (date).
        """
        # Get the current date corresponding to the step
        current_date = self.df.index.get_level_values(0).unique()[step]

        # Filter the DataFrame for this date (all tickers)
        current_data = self.df.loc[current_date]

        # Extract the 'Close' prices for all tickers on the current date
        close_prices = current_data['Close']

        return close_prices

    def _take_action(self, action, current_prices):
        """
        Execute the action (rebalancing), update the number of shares (self.holdings),
        and adjust the cash balance (self.balance), considering transaction fees.

        Args:
        - action: The target portfolio weights provided by the agent.
        - current_prices: The current close prices of the stocks.
        """
        # Normalize the action
        action = self._normalize_action(action)

        # Target cash allocation and stock allocations (dollar amounts)
        target_stock_allocations = action[1:] * self.portfolio_value

        # Current stock allocations (dollar amounts)
        current_stock_allocations = self.holdings * current_prices

        # Deltas represent changes in stock allocations (buy/sell)
        deltas = target_stock_allocations - current_stock_allocations

        # Calculate transaction fees
        transaction_fees = np.abs(deltas) * self.transaction_fee_rate
        self._transaction_fees.append(np.sum(transaction_fees))
        

        # Calculate net cash flow from trades
        net_cash_flow = np.sum(deltas) # Positive: cash needed, Negative: cash generated from selling

        # Total cost includes transaction fees and cash required for buys
        total_transaction_cost = np.sum(transaction_fees) + net_cash_flow

        if total_transaction_cost > self.balance:
            # If there's insufficient cash, reject the action
            # print(f"Insufficient funds. Transaction cost: {total_transaction_cost}, Available cash: {self.balance}")
            return

        # Update holdings (new number of shares for each stock)
        self.holdings = target_stock_allocations / current_prices

        # Update cash balance (account for transaction fees and cash spent/received)
        self.balance -= total_transaction_cost

        # Debug information (optional)
        # print(f"Action taken: {action}")
        # print(f"Target stock allocations: {target_stock_allocations}")
        # print(f"Delta in stock allocations: {deltas}")
        # print(f"Transaction fees: {transaction_fees}")
        # print(f"New cash balance: {self.balance}")
        # print(f"New cash balance: {self.holdings}")
        # print(f"New portfolio weights: {self.portfolio_weights}")

    def _update_portfolio_value(self, current_prices):
        """
        Update the portfolio value based on current stock prices and holdings, and update portfolio weights.

        Args:
        - current_prices: The current close prices of the stocks.
        """
        # Step 1: Calculate the value of stock holdings (number of shares * current prices)
        stock_values = np.sum(self.holdings * current_prices)

        # Step 2: Update the total portfolio value (stock values + cash balance)
        self.portfolio_value = self.balance + stock_values

        # Step 3: Update the portfolio weights based on the new portfolio value
        #         All weights sum up to 1
        self.portfolio_weights[1:] = (self.holdings * current_prices) / self.portfolio_value  # Stock weights
        self.portfolio_weights[0] = self.balance / self.portfolio_value  # Cash weight


    def _calculate_reward(self, previous_portfolio_value=None):
        """
        Calculate the reward based on the change in portfolio value.
        """
        # Use one-step simple return to keep reward scale independent of episode length.
        if previous_portfolio_value is None or previous_portfolio_value <= 0:
            return 0.0

        reward = (self.portfolio_value / previous_portfolio_value) - 1.0

        return reward


    def _get_observation(self):
        """
        Returns the observation, which includes:
        - Market data: OHLCV, technical indicators, and covariances for all stocks.
        - Portfolio weights and cash balance.
        """
        # Get the current date's data (OHLCV, technical indicators, covariances) for all tickers
        current_date = self.df.index.get_level_values(0).unique()[self.current_step]
        current_data = self.df.loc[current_date]

        # Combine OHLCV, technical indicators, and covariances
        market_data = current_data.values.flatten()
        obs = np.concatenate([
            market_data,
            self.portfolio_weights,
            [self.balance]
        ])

        return obs

    def render(self, mode='human'):
        """
        Render the environment.
        Output useful information like portfolio value, holdings, stock prices, etc.
        """
        if self.current_step >= len(self.df.index.get_level_values(0).unique()) - 1:
            # Calculate and print the portfolio metrics
            metrics = self._calculate_portfolio_metrics()

            print(f"End of Episode Metrics:")
            print(f"Cumulative Return: {metrics['Cumulative Return']:.2f}")
            print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown: {metrics['Max Drawdown']:.2f}")
            print(f"Volatility: {metrics['Volatility']:.2f}")
            print("="*50)

        # Get the current date corresponding to the current step
        current_date = self.df.index.get_level_values(0).unique()[self.current_step]

        # Get the close prices for the current step
        current_prices = self._get_close_prices(self.current_step)

        # Display key information
        print(f"Step: {self.current_step}")
        print(f"Date: {current_date}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Cash Balance: {self.balance:.2f}")
        print(f"Stock Prices: {current_prices}")
        print(f"Portfolio Weights: {self.portfolio_weights}")
        print("="*50)

    def _calculate_portfolio_metrics(self):
        """
        Calculate portfolio performance metrics like Sharpe Ratio, Max Drawdown, and Volatility.
        - Cumulative return is based on the final value of the portfolio.
        - Daily returns are calculated from the change in portfolio value between steps.

        Returns:
        - Dictionary with portfolio performance metrics.
        """
        # Convert cumulative rewards to portfolio values
        portfolio_values = np.array(self._portfolio_value_memory)

        # 1. Cumulative Return: Final portfolio value vs. initial balance
        cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1

        # 2. Daily Returns: Compute daily returns from portfolio values
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # 3. Sharpe Ratio: Mean return divided by the standard deviation of returns
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe_ratio = mean_return / (std_return + 1e-6)  # Avoid division by zero

        # 4. Max Drawdown: The maximum drop from peak portfolio value to trough
        peak_value = np.maximum.accumulate(portfolio_values)
        drawdown = (peak_value - portfolio_values) / peak_value
        max_drawdown = np.max(drawdown)

        # 5. Volatility: Standard deviation of daily returns
        volatility = std_return

        return {
            'Cumulative Return': cumulative_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Volatility': volatility
        }
    
    def log_episode(self):
        """
        Save the details of the episode, including actions, portfolio values, and other metrics.
        """

        # Actions at decision date t are executed/valued at date t+1.
        start = self.episode_start_step + 1
        end = start + len(self._action_memory)
        dates = self.trading_dates[start:end]

        episode_log = {
            'Episode': [self.episode] * len(self._action_memory),
            'Dates': dates,
            'Actions': self._action_memory,
            'Portfolio Values': self._portfolio_value_memory[1:],
            'Transaction Costs': self._transaction_fees,
            'Weights': self._final_weights_memory[1:],
        }

        # Convert to DataFrame
        log_df = pd.DataFrame(episode_log)

        if self.logfile:
            # Append the log to CSV (use mode='a' to append, header=False if the file already exists)
            log_df.to_csv(self.logfile, mode='a', header=not pd.io.common.file_exists(self.logfile), index=False)

        return episode_log

class PortfolioAllocationEnvLogReturn(PortfolioAllocationEnv):
    """
    Inherits from PortfolioAllocationEnv and overrides _calculate_reward()
    to return the daily log return of the portfolio.
    """
    def _calculate_reward(self, previous_portfolio_value=None):
        """
        Calculate the daily log return based on the portfolio value.
        """
        if previous_portfolio_value is None or previous_portfolio_value <= 0:
            return 0

        # Calculate log return
        log_return = np.log(self.portfolio_value / previous_portfolio_value)

        return log_return
