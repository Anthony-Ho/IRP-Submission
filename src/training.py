import os
from stable_baselines3 import PPO, A2C, DDPG

#Local Import
from envs import PortfolioAllocationEnv, PortfolioAllocationEnvLogReturn
from continual_learning import *
from performance import validate_agent_performance

def train_with_early_stopping(agent, train_env, val_env, total_timesteps, validation_interval, patience, model_dir):
    """
    Train the agent with early stopping based on validation performance.
    """
    best_val_performance = -float('inf')
    best_model = None
    epochs_without_improvement = 0
    timesteps_per_interval = total_timesteps // validation_interval

    for i in range(validation_interval):
        # Train for a set of timesteps
        agent.learn(total_timesteps=timesteps_per_interval)

        # Evaluate the agent on the validation environment
        val_performance = validate_agent_performance(agent, val_env)

        # Check for improvement
        if val_performance > best_val_performance:
            best_val_performance = val_performance
            best_model = agent  # Save best model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print(f"Early stopping at iteration {i}. Best validation performance: {best_val_performance}")
            break

    # Save the best model after early stopping
    # best_model.save(f"{model_dir}/best_model.zip")
    return best_model

# Updated function to train PPO, A2C, and DDPG agents with early stopping if validation_df is provided
def train_baseline_agents(model_dir, train_df1, group1, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, 
                          transaction_fee_rate=0.001, initial_balance=100000, validation_df=None, 
                          validation_interval=10, patience=3, total_timesteps=[50000, 80000, 50000], env_class=PortfolioAllocationEnv):
    """
    Train PPO, A2C, and DDPG agents on group1 data and save the models.
    If validation_df is provided, use early stopping based on validation performance.

    Parameters:
    - model_dir: The directory to save the trained models.
    - train_df1: DataFrame containing group1 data for training.
    - group1: List of tickers in group1.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - transaction_fee_rate: Transaction fee rate for the environment.
    - initial_balance: Initial balance for the portfolio.
    - validation_df: DataFrame for validation (optional).
    - validation_interval: Number of validation steps between each interval (for early stopping).
    - patience: Early stopping patience (number of intervals without improvement before stopping).
    - total_timesteps: Total timesteps for training.

    Returns:
    - ppo_model, a2c_model, ddpg_model: Trained PPO, A2C, and DDPG models.
    """
    
    # Create a unique group name based on the iteration number
    group_name = f"baseline_{iteration}"
    
    # Create the training environment for group1 data
    train_env = env_class(df=train_df1, initial_balance=initial_balance, tic_list=group1, transaction_fee_rate=transaction_fee_rate)

    # If validation_df is provided, create the validation environment
    val_env = None
    if validation_df is not None:
        val_env = env_class(df=validation_df, initial_balance=initial_balance, tic_list=group1, transaction_fee_rate=transaction_fee_rate)

    # Train PPO agent
    ppo_model = PPO("MlpPolicy", train_env, verbose=1, **PPO_PARAMS)
    if val_env is not None:
        ppo_model = train_with_early_stopping(ppo_model, train_env, val_env, total_timesteps[0], validation_interval, patience, model_dir)
    else:
        ppo_model.learn(total_timesteps=total_timesteps[0])  # Adjust timesteps if needed
    ppo_model.save(os.path.join(model_dir, f"ppo_{group_name}"))
    
    # Train A2C agent
    a2c_model = A2C("MlpPolicy", train_env, verbose=1, **A2C_PARAMS)
    if val_env is not None:
        a2c_model = train_with_early_stopping(a2c_model, train_env, val_env, total_timesteps[1], validation_interval, patience, model_dir)
    else:
        a2c_model.learn(total_timesteps=total_timesteps[1])  # Adjust timesteps if needed
    a2c_model.save(os.path.join(model_dir, f"a2c_{group_name}"))
    
    # Train DDPG agent
    ddpg_model = DDPG("MlpPolicy", train_env, verbose=1, **DDPG_PARAMS)
    if val_env is not None:
        ddpg_model = train_with_early_stopping(ddpg_model, train_env, val_env, total_timesteps[2], validation_interval, patience, model_dir)
    else:
        ddpg_model.learn(total_timesteps=total_timesteps[2])  # Adjust timesteps if needed
    ddpg_model.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return trained models
    return ppo_model, a2c_model, ddpg_model

def train_naive_strategy(model_dir, train_df2, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, 
                         transaction_fee_rate=0.001, initial_balance=100000, validation_df=None, 
                         validation_interval=10, patience=3, total_timesteps=[50000, 80000, 50000], 
                         env_class=PortfolioAllocationEnv):
    """
    Train PPO, A2C, and DDPG agents using the naive continual learning strategy.
    If validation_df is provided, use early stopping based on validation performance.
    Each agent can be trained with a different number of timesteps.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df2: DataFrame containing group2 data for training.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - transaction_fee_rate: Transaction fee rate for the environment.
    - initial_balance: Initial balance for the portfolio.
    - validation_df: DataFrame for validation (optional).
    - validation_interval: Number of validation steps between each interval (for early stopping).
    - patience: Early stopping patience (number of intervals without improvement before stopping).
    - total_timesteps: List of timesteps for PPO, A2C, and DDPG agents [PPO_timesteps, A2C_timesteps, DDPG_timesteps].
    - env_class: The environment class to be used (default is PortfolioAllocationEnv for absolute returns).

    Returns:
    - naive_ppo, naive_a2c, naive_ddpg: Trained PPO, A2C, and DDPG models.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"naive_{iteration}"

    # Create the training environment for group2 using the specified environment class
    group2_env = env_class(df=train_df2, initial_balance=initial_balance, tic_list=group2, transaction_fee_rate=transaction_fee_rate)

    # If validation_df is provided, create the validation environment using the same class
    val_env = None
    if validation_df is not None:
        val_env = env_class(df=validation_df, initial_balance=initial_balance, tic_list=group2, transaction_fee_rate=transaction_fee_rate)

    # Train PPO agent using naive strategy
    naive_ppo = PPO.load(os.path.join(model_dir, f"ppo_{baseline_name}"), env=group2_env)
    if val_env is not None:
        naive_ppo = train_with_early_stopping(naive_ppo, group2_env, val_env, total_timesteps[0], validation_interval, patience, model_dir)
    else:
        naive_ppo.learn(total_timesteps=total_timesteps[0])
    naive_ppo.save(os.path.join(model_dir, f"ppo_{group_name}"))

    # Train A2C agent using naive strategy
    naive_a2c = A2C.load(os.path.join(model_dir, f"a2c_{baseline_name}"), env=group2_env)
    if val_env is not None:
        naive_a2c = train_with_early_stopping(naive_a2c, group2_env, val_env, total_timesteps[1], validation_interval, patience, model_dir)
    else:
        naive_a2c.learn(total_timesteps=total_timesteps[1])
    naive_a2c.save(os.path.join(model_dir, f"a2c_{group_name}"))

    # Train DDPG agent using naive strategy
    naive_ddpg = DDPG.load(os.path.join(model_dir, f"ddpg_{baseline_name}"), env=group2_env)
    if val_env is not None:
        naive_ddpg = train_with_early_stopping(naive_ddpg, group2_env, val_env, total_timesteps[2], validation_interval, patience, model_dir)
    else:
        naive_ddpg.learn(total_timesteps=total_timesteps[2])
    naive_ddpg.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return the trained agents
    return naive_ppo, naive_a2c, naive_ddpg


def train_ewc_agents(model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS, 
                     transaction_fee_rate=0.001, initial_balance=100000, validation_df=None, 
                     validation_interval=10, patience=3, total_timesteps=[50000, 80000, 50000], 
                     lambda_ewc=0.4, env_class=PortfolioAllocationEnv):
    """
    Train PPO, A2C, and DDPG agents using the EWC strategy.
    If validation_df is provided, use early stopping based on validation performance.
    Each agent can be trained with a different number of timesteps.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df1: DataFrame containing group1 data for calculating Fisher Information matrix.
    - train_df2: DataFrame containing group2 data for training with EWC.
    - group1: List of tickers in group1.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - transaction_fee_rate: Transaction fee rate for the environment.
    - initial_balance: Initial balance for the portfolio.
    - validation_df: DataFrame for validation (optional).
    - validation_interval: Number of validation steps between each interval (for early stopping).
    - patience: Early stopping patience (number of intervals without improvement before stopping).
    - total_timesteps: List of timesteps for PPO, A2C, and DDPG agents [PPO_timesteps, A2C_timesteps, DDPG_timesteps].
    - lambda_ewc: Regularization strength for EWC.
    - env_class: The environment class to be used (default is PortfolioAllocationEnv for absolute returns).

    Returns:
    - ewc_ppo_agent, ewc_a2c_agent, ewc_ddpg_agent: Trained PPO, A2C, and DDPG models using EWC.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"ewc_{iteration}"

    # Create the training environments for group1 and group2 using the specified environment class
    group1_env = env_class(df=train_df1, initial_balance=initial_balance, tic_list=group1, transaction_fee_rate=transaction_fee_rate)
    group2_env = env_class(df=train_df2, initial_balance=initial_balance, tic_list=group2, transaction_fee_rate=transaction_fee_rate)

    # If validation_df is provided, create the validation environment
    val_env = None
    if validation_df is not None:
        val_env = env_class(df=validation_df, initial_balance=initial_balance, tic_list=group2, transaction_fee_rate=transaction_fee_rate)

    ## PPO Agent with EWC ##
    ppo_model_group1 = PPO.load(os.path.join(model_dir, f"ppo_{baseline_name}"))

    # Create DataLoader and initialize EWC for PPO
    train_group1_dataloader = create_dataloader(
        agent=ppo_model_group1,
        env=group1_env,
        desired_num_observations=len(train_df1),
        batch_size=64,
        use_agent_policy=True
    )
    ewc_ppo = EWC(agent=ppo_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_PPO agent
    ewc_ppo_agent = EWC_PPO(
        policy=ppo_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_ppo,
        verbose=1
    )

    # Load pre-trained policy weights and train the EWC agent
    ewc_ppo_agent.policy.load_state_dict(ppo_model_group1.policy.state_dict())
    if val_env is not None:
        ewc_ppo_agent = train_with_early_stopping(ewc_ppo_agent, group2_env, val_env, total_timesteps[0], validation_interval, patience, model_dir)
    else:
        ewc_ppo_agent.learn(total_timesteps=total_timesteps[0])
    ewc_ppo_agent.save(os.path.join(model_dir, f"ppo_{group_name}"))

    ## A2C Agent with EWC ##
    a2c_model_group1 = A2C.load(os.path.join(model_dir, f"a2c_{baseline_name}"))

    # Create DataLoader and initialize EWC for A2C
    train_group1_dataloader = create_dataloader(
        agent=a2c_model_group1,
        env=group1_env,
        desired_num_observations=len(train_df1),
        batch_size=64,
        use_agent_policy=True
    )
    ewc_a2c = EWC(agent=a2c_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_A2C agent
    ewc_a2c_agent = EWC_A2C(
        policy=a2c_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_a2c,
        verbose=1
    )

    # Load pre-trained policy weights and train the EWC agent
    ewc_a2c_agent.policy.load_state_dict(a2c_model_group1.policy.state_dict())
    if val_env is not None:
        ewc_a2c_agent = train_with_early_stopping(ewc_a2c_agent, group2_env, val_env, total_timesteps[1], validation_interval, patience, model_dir)
    else:
        ewc_a2c_agent.learn(total_timesteps=total_timesteps[1])
    ewc_a2c_agent.save(os.path.join(model_dir, f"a2c_{group_name}"))

    ## DDPG Agent with Deterministic EWC ##
    ddpg_model_group1 = DDPG.load(os.path.join(model_dir, f"ddpg_{baseline_name}"))

    # Create DataLoader for DDPG and initialize Deterministic EWC for the actor network
    train_group1_dataloader = create_dataloader(
        agent=ddpg_model_group1,
        env=group1_env,
        desired_num_observations=len(train_df1),
        batch_size=64,
        use_agent_policy=True
    )
    ewc_ddpg = Deterministic_EWC(agent=ddpg_model_group1, dataloader=train_group1_dataloader, lambda_=lambda_ewc)

    # Create the EWC_DDPG agent and load the pre-trained networks
    ewc_ddpg_agent = EWC_DDPG(
        policy=ddpg_model_group1.policy.__class__,
        env=group2_env,
        ewc=ewc_ddpg,
        verbose=1,
    )

    ewc_ddpg_agent.actor.load_state_dict(ddpg_model_group1.actor.state_dict())
    ewc_ddpg_agent.critic.load_state_dict(ddpg_model_group1.critic.state_dict())
    ewc_ddpg_agent.actor_target.load_state_dict(ddpg_model_group1.actor_target.state_dict())
    ewc_ddpg_agent.critic_target.load_state_dict(ddpg_model_group1.critic_target.state_dict())

    # Train the EWC_DDPG agent
    if val_env is not None:
        ewc_ddpg_agent = train_with_early_stopping(ewc_ddpg_agent, group2_env, val_env, total_timesteps[2], validation_interval, patience, model_dir)
    else:
        ewc_ddpg_agent.learn(total_timesteps=total_timesteps[2])
    ewc_ddpg_agent.save(os.path.join(model_dir, f"ddpg_{group_name}"))

    # Return trained agents
    return ewc_ppo_agent, ewc_a2c_agent, ewc_ddpg_agent


# Function to train PPO, A2C, and DDPG agents using the Buffer Replay strategy
def train_replay_agents(model_dir, train_df1, train_df2, group1, group2, iteration, PPO_PARAMS, A2C_PARAMS, DDPG_PARAMS,
                         transaction_fee_rate=0.001, initial_balance=100000, validation_df=None, 
                         validation_interval=10, patience=3, total_timesteps=[50000, 80000, 50000], 
                         env_class=PortfolioAllocationEnv):
    """
    Train PPO, A2C, and DDPG agents using the Buffer Replay strategy.

    Parameters:
    - model_dir: The directory where the pretrained models are saved.
    - train_df1: DataFrame containing group1 data for calculating Fisher Information matrix.
    - train_df2: DataFrame containing group2 data for training with Buffer Replay.
    - group1: List of tickers in group1.
    - group2: List of tickers in group2.
    - iteration: The iteration number (int) for naming the model files.
    - PPO_PARAMS: Hyperparameters for training the PPO agent.
    - A2C_PARAMS: Hyperparameters for training the A2C agent.
    - DDPG_PARAMS: Hyperparameters for training the DDPG agent.
    - lambda_ewc: Regularization strength for Buffer Replay.
    - transaction_fee_rate: Transaction fee rate for the environment.
    - initial_balance: Initial balance for the portfolio.
    - validation_df: DataFrame for validation (optional).
    - validation_interval: Number of validation steps between each interval (for early stopping).
    - patience: Early stopping patience (number of intervals without improvement before stopping).
    - total_timesteps: List of timesteps for PPO, A2C, and DDPG agents [PPO_timesteps, A2C_timesteps, DDPG_timesteps].
    - lambda_ewc: Regularization strength for EWC.
    - env_class: The environment class to be used (default is PortfolioAllocationEnv for absolute returns).

    Returns:
    - replay_ppo, replay_a2c, replay_ddpg: Trained PPO, A2C, and DDPG models using Buffer Replay.
    """
    
    # Create a unique group name based on the iteration number
    baseline_name = f"baseline_{iteration}"
    group_name = f"replay_{iteration}"

    replay_ppo = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=total_timesteps[0],
        agent_class=PPO,
        agent_filename=f"ppo_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"ppo_{group_name}",
        params=PPO_PARAMS,
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=transaction_fee_rate,
        initial_balance=initial_balance,
        validation_df=validation_df,
        validation_interval=validation_interval,
        patience=patience,
        env_class=env_class
    )

    replay_a2c = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=total_timesteps[1],
        agent_class=A2C,
        agent_filename=f"a2c_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"a2c_{group_name}",
        params=A2C_PARAMS,
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=transaction_fee_rate,
        initial_balance=initial_balance,
        validation_df=validation_df,
        validation_interval=validation_interval,
        patience=patience,
        env_class=env_class
    )

    replay_ddpg = perform_replay_training(
        train_df1=train_df1,
        train_df2=train_df2,
        tic_list_group1=group1,
        tic_list_group2=group2,
        total_timesteps=total_timesteps[2],
        agent_class=DDPG,
        agent_filename=f"ddpg_{baseline_name}",
        model_dir=model_dir,
        model_filename=f"ddpg_{group_name}",
        params=DDPG_PARAMS,
        reinforcement_interval=15000,
        reinforcement_steps=3000,
        transaction_fee_rate=transaction_fee_rate,
        initial_balance=initial_balance,
        validation_df=validation_df,
        validation_interval=validation_interval,
        patience=patience,
        env_class=env_class
    )

    return replay_ppo, replay_a2c, replay_ddpg