import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import polyak_update, explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn.functional as F
import gymnasium as gym
import random
import os
from gymnasium import spaces

# Local Import
from envs import PortfolioAllocationEnv
from performance import validate_agent_performance

def create_dataloader(agent, env, desired_num_observations=10000, batch_size=64, use_agent_policy=True):
    """
    Creates a DataLoader from observations collected using the agent in the given environment.

    Parameters:
    - agent: The pre-trained agent used to collect observations.
    - env: The environment from which to collect observations.
    - desired_num_observations: The number of observations to collect.
    - batch_size: The batch size for the DataLoader.
    - use_agent_policy: If True, use the agent's policy to select actions; if False, use random actions.

    Returns:
    - dataloader: A DataLoader containing the observations.
    """
    # Initialize storage for observations
    observations = []
    
    # Reset the environment to start collecting data
    obs, _ = env.reset()
    done = False
    steps = 0  # Counter for steps in the current episode

    while True:
        observations.append(obs)
        if use_agent_policy:
            # Use the agent's policy to select an action
            action, _ = agent.predict(obs)
        else:
            # Use a random action
            action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if done or truncated:
            obs, _ = env.reset()
            steps = 0  # Reset step counter for new episode
        if len(observations) >= desired_num_observations:
            break
    
    # Convert observations to a PyTorch tensor
    observations_array = np.array(observations)
    if len(observations_array.shape) == 1:
        observations_array = np.expand_dims(observations_array, axis=-1)
    observations_tensor = torch.tensor(observations_array, dtype=torch.float32)
    
    # Move tensor to the same device as the agent
    observations_tensor = observations_tensor.to(agent.device)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(observations_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

class EWC:
    def __init__(self, agent, dataloader, lambda_=0.4):
        self.agent = agent
        self.lambda_ = lambda_
        self.params = {n: p.clone().detach() for n, p in agent.policy.named_parameters() if p.requires_grad}
        self.fisher_matrix = self.calculate_fisher_information(dataloader)
    
    def calculate_fisher_information(self, dataloader):
        fisher_matrix = {}
        for n, p in self.params.items():
            fisher_matrix[n] = torch.zeros_like(p)

        for data in dataloader:
            obs = data[0]  # Corrected indexing
            # Ensure observations are on the correct device
            obs = obs.to(self.agent.device)
            
            # Move obs to CPU and convert to NumPy array
            obs_numpy = obs.cpu().numpy()
    
            action, _ = self.agent.policy.predict(obs_numpy, deterministic=True)
            action_tensor = torch.tensor(action).to(self.agent.device)
            
            # Forward pass through the policy network
            log_prob = self.agent.policy.evaluate_actions(obs, action_tensor)[1]
            loss = -log_prob.mean()  # Maximize log probability
            self.agent.policy.optimizer.zero_grad()
            loss.backward()

            # Accumulate Fisher Information
            for n, p in self.agent.policy.named_parameters():
                if p.grad is not None:
                    fisher_matrix[n] += p.grad ** 2

        # Normalize Fisher Information
        for n in fisher_matrix:
            fisher_matrix[n] /= len(dataloader)
        return fisher_matrix

    def penalty(self):
        penalty_loss = 0
        for n, p in self.agent.policy.named_parameters():
            if p.requires_grad:
                penalty_loss += (self.fisher_matrix[n] * (p - self.params[n]) ** 2).sum()
        return (self.lambda_ / 2) * penalty_loss


class EWC_PPO(PPO):
    def __init__(self, *args, ewc=None, **kwargs):
        super(EWC_PPO, self).__init__(*args, **kwargs)
        self.ewc = ewc

    def train(self) -> None:
        """
        Modified train() method based on stable-baselines3 implementation.

        This method is adapted from the original train() method of stable-baselines3
        (version X.X.X). The primary modification is the addition of the EWC penalty
        to the loss calculation, which is applied as part of the Elastic Weight Consolidation
        (EWC) technique to preserve knowledge from previous tasks.

        Source: https://github.com/DLR-RM/stable-baselines3
        """
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Add EWC penalty
                if self.ewc is not None:
                    ewc_penalty = self.ewc.penalty()
                    loss += ewc_penalty

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class EWC_A2C(A2C):
    def __init__(self, *args, ewc=None, **kwargs):
        super(EWC_A2C, self).__init__(*args, **kwargs)
        self.ewc = ewc

    def train(self) -> None:
        """
        Modified train() method based on stable-baselines3 implementation.

        This method is adapted from the original train() method of stable-baselines3
        (version X.X.X). The primary modification is the addition of the EWC penalty
        to the loss calculation, which is applied as part of the Elastic Weight Consolidation
        (EWC) technique to preserve knowledge from previous tasks.

        Source: https://github.com/DLR-RM/stable-baselines3
        """
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        ewc_penalties = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Add EWC penalty
            if self.ewc is not None:
                ewc_penalty = self.ewc.penalty()
                loss += ewc_penalty
                ewc_penalties.append(ewc_penalty.item())

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        if self.ewc is not None:
            self.logger.record("train/ewc_penalty", np.mean(ewc_penalties))


class Deterministic_EWC:
    def __init__(self, agent, dataloader, lambda_=0.4):
        self.agent = agent  # The pre-trained agent
        self.lambda_ = lambda_
        # Store the parameters of the actor network
        self.params = {n: p.clone().detach() for n, p in agent.actor.named_parameters() if p.requires_grad}
        self.fisher_matrix = self.calculate_fisher_information(dataloader)

    def calculate_fisher_information(self, dataloader):
        fisher_matrix = {}
        for n, p in self.params.items():
            fisher_matrix[n] = torch.zeros_like(p)

        for data in dataloader:
            obs = data[0].to(self.agent.device)
            # Get actions from the pre-trained actor
            actions = self.agent.actor(obs)
            # Do not detach actions
            # Optionally add small noise to simulate stochasticity
            noise = torch.normal(0, 1e-6, size=actions.shape).to(actions.device)
            actions_noisy = actions + noise
            # Compute approximate log probabilities
            log_probs = -0.5 * ((actions_noisy) ** 2).sum(dim=1)
            loss = -log_probs.mean()
            self.agent.actor.zero_grad()
            loss.backward()

            # Accumulate Fisher Information
            for n, p in self.agent.actor.named_parameters():
                if p.grad is not None:
                    fisher_matrix[n] += p.grad.data.clone() ** 2

        # Normalize Fisher Information
        for n in fisher_matrix:
            fisher_matrix[n] /= len(dataloader)
        return fisher_matrix

    def penalty(self, network):
        penalty_loss = 0
        for n, p in network.named_parameters():
            if p.requires_grad:
                penalty_loss += (self.fisher_matrix[n] * (p - self.params[n]) ** 2).sum()
        return (self.lambda_ / 2) * penalty_loss


class EWC_DDPG(DDPG):
    def __init__(self, *args, ewc=None, **kwargs):
        super(EWC_DDPG, self).__init__(*args, **kwargs)
        self.ewc = ewc

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Modified train() method based on stable-baselines3 implementation.

        This method is adapted from the original train() method of stable-baselines3
        (version X.X.X). The primary modification is the addition of the EWC penalty
        to the loss calculation, which is applied as part of the Elastic Weight Consolidation
        (EWC) technique to preserve knowledge from previous tasks.

        Source: https://github.com/DLR-RM/stable-baselines3
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, ewc_penalties = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                
                # Add EWC penalty to actor loss
                if self.ewc is not None:
                    ewc_penalty = self.ewc.penalty(self.actor)
                    actor_loss += ewc_penalty
                    ewc_penalties.append(ewc_penalty.item())

                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


# Function to create a new environment with randomly selected 1-year data from group1
def create_group1_env_random(group1_df, tic_list, num_days=252, transaction_fee_rate=0.001, initial_balance=100000,
                             env_class=PortfolioAllocationEnv):
    group1_df = group1_df.sort_index()
    unique_date = group1_df.index.get_level_values('Date').unique()
    # Ensure there are at least 252 days available
    if len(unique_date) < num_days:
        raise ValueError("Insufficient data for group1.")

    # Randomly select a start date such that 252 days of data are available
    start_idx = random.randint(0, len(unique_date) - num_days)
    end_idx = start_idx + num_days
    selected_dates = unique_date[start_idx:end_idx]

    # Extract the 1-year subset of group1 data
    group1_subset = group1_df.loc[(slice(selected_dates[0], selected_dates[-1]), slice(None))]

    # Create a new environment with the selected data subset
    group1_env = env_class(
        df=group1_subset, 
        initial_balance=initial_balance, 
        tic_list=tic_list, 
        transaction_fee_rate=transaction_fee_rate
    )
    
    # Wrap the environment if necessary (e.g., with DummyVecEnv for compatibility)
    # group1_env = DummyVecEnv([lambda: group1_env])
    
    return group1_env

# Function to create the group2 environment
def create_group2_env(train_df2, tic_list, transaction_fee_rate=0.001, initial_balance=100000,
                      env_class=PortfolioAllocationEnv):
    group2_env = env_class(
        df=train_df2, 
        initial_balance=initial_balance, 
        tic_list=tic_list, 
        transaction_fee_rate=transaction_fee_rate
    )
    
    # Wrap the environment if necessary
    # group2_env = DummyVecEnv([lambda: group2_env])
    
    return group2_env

from datetime import datetime

# Main function to perform training and save the model
def perform_replay_training(train_df1, train_df2, tic_list_group1, tic_list_group2, total_timesteps, 
                     agent_class, agent_filename, model_dir, model_filename, params,
                    reinforcement_interval=15000, reinforcement_steps=3000, transaction_fee_rate=0.001, initial_balance=100000,
                    validation_df=None, validation_interval=10, patience=3, env_class=PortfolioAllocationEnv):
    """
    Trains an RL agent on group2 data with periodic reinforcement on group1 data.

    Parameters:
    - train_df1 (pd.DataFrame): Training data for group1.
    - train_df2 (pd.DataFrame): Training data for group2.
    - tic_list_group1 (list): List of tickers/assets for group1.
    - tic_list_group2 (list): List of tickers/assets for group2.
    - total_timesteps (int): Total number of timesteps for training.
    - agent_class (class): RL agent class from Stable Baselines3 (e.g., DDPG, A2C, PPO).
    - model_dir (str): Directory to save the trained models.
    - params: parameter for agent model.
    - reinforcement_interval (int): Timesteps to train on group2 before reinforcement.
    - reinforcement_steps (int): Timesteps to train on group1 during reinforcement.
    - transaction_fee_rate (float): Transaction fee rate for the environment.
    - initial_balance (int): Initial balance for the portfolio.
    - env_class: The environment class to be used (default is PortfolioAllocationEnv for absolute returns).

    Returns:
    - The trained agent.
    """
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    best_val_performance = -float('inf')
    best_model = None
    epochs_without_improvement = 0
    early_stop = False

    # Create the group2 environment
    group2_env = create_group2_env(train_df2, tic_list_group2, transaction_fee_rate, initial_balance, 
                                   env_class=env_class)
    
    # f validation_df is provided, create the validation environment
    val_env = None
    if validation_df is not None:
        val_env = env_class(df=validation_df, initial_balance=initial_balance, tic_list=tic_list_group2, transaction_fee_rate=transaction_fee_rate)

    # Initialize the agent with group2 environment
    group1_model_path = os.path.join(model_dir, agent_filename)
    agent = agent_class.load(group1_model_path, env=group2_env)
    timesteps_between_validation = total_timesteps // validation_interval

    # Step 1: Create model switching milestones
    model_milestones = []
    timesteps_done = 0
    while timesteps_done < total_timesteps:
        # Add a milestone for training on group2
        model_milestones.append((timesteps_done, "model2"))
        # Add a milestone for training on group1 after reinforcement_interval
        timesteps_done += reinforcement_interval
        if timesteps_done < total_timesteps:
            model_milestones.append((timesteps_done, "model1"))
            # Train on group1 for reinforcement_steps
            timesteps_done += reinforcement_steps

    # Step 2: Create validation milestones 
    # Validation starts only at least one Group1 memory is replayed.
    if val_env is not None:
        validation_milestones = []
        for t in range(reinforcement_interval+reinforcement_steps, total_timesteps, timesteps_between_validation):
            validation_milestones.append((t, "validation"))

        # Step 3: Merge and sort milestones by timestep
        milestones = sorted(model_milestones + validation_milestones, key=lambda x: x[0])
    else:
        milestones = model_milestones

    # Step 4: Go thru the milestone interlace training and validation
    for i in range(len(milestones)):
        current_timestep, action = milestones[i]
        if (i == (len(milestones)-1)):
            next_timestep = total_timesteps
        else:
            next_timestep, _ = milestones[i + 1]
        timesteps_to_learn = next_timestep - current_timestep

        # Take the action either change envs or validataion
        if (action == 'validation'):
            print(f"Do Validation at {current_timestep}")
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
                early_stop = True
                print(f"Early stopping at iteration {current_timestep}. Best validation performance: {best_val_performance}")
                break
        elif (action == 'model2'):
            print(f"Switching back to group2 environment at timestep {current_timestep}.")
            agent.set_env(group2_env)
        elif (action == 'model1'):
            # Create a random group1 environment for reinforcement
            print(f"Switching to group1 environment at timestep {current_timestep}.")
            try:
                random_group1_env = create_group1_env_random(
                    train_df1, 
                    tic_list_group1,
                    num_days=int(reinforcement_steps/10), 
                    transaction_fee_rate=transaction_fee_rate, 
                    initial_balance=initial_balance,
                    env_class=env_class
                )
            except ValueError as e:
                print(f"Error creating group1 environment: {e}")
                break

            # Set the new environment for the agent
            agent.set_env(random_group1_env)

        # Train between the current and next milestone
        print(f"Training for {timesteps_to_learn} timesteps.")
        if (current_timestep == 0):
            agent.learn(total_timesteps=timesteps_to_learn, reset_num_timesteps=True)
        elif (timesteps_to_learn > 0):
            agent.learn(total_timesteps=timesteps_to_learn, reset_num_timesteps=False)

    # Final model saving
    final_model_path = os.path.join(model_dir, model_filename)
    if early_stop:
        best_model.save(final_model_path)
        print(f"Early Stopped model saved to {final_model_path}")
        return best_model
    else:
        agent.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        return agent