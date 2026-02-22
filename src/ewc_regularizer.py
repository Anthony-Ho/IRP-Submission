import torch


class EWC:
    """
    Diagonal Elastic Weight Consolidation (EWC) regularizer for actor-critic policies.

    This class snapshots a reference policy's parameters and estimates a diagonal
    Fisher Information matrix from observations. During later training, `penalty()`
    can be evaluated on a target policy to discourage drift on important weights.
    """

    def __init__(self, agent, dataloader, lambda_=0.4):
        self.lambda_ = lambda_
        self.device = agent.device
        self.reference_params = {
            name: param.detach().clone()
            for name, param in agent.policy.named_parameters()
            if param.requires_grad
        }
        self.fisher_diagonal = self._estimate_fisher(agent, dataloader)

    def _estimate_fisher(self, agent, dataloader):
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.reference_params.items()
        }

        num_batches = 0
        for data in dataloader:
            obs = data[0].to(self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # Compute score-function gradients from sampled actions.
            distribution = agent.policy.get_distribution(obs)
            sampled_actions = distribution.get_actions(deterministic=False)
            log_prob = distribution.log_prob(sampled_actions)
            loss = -log_prob.mean()

            agent.policy.optimizer.zero_grad()
            loss.backward()

            for name, param in agent.policy.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            num_batches += 1

        if num_batches == 0:
            return fisher

        for name in fisher:
            fisher[name] /= num_batches

        return fisher

    def penalty(self, target_policy):
        # Keep this as a scalar tensor to avoid shape/broadcast issues
        # when adding to scalar policy losses.
        penalty_loss = torch.zeros((), device=self.device)

        for name, param in target_policy.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.reference_params:
                continue

            param_ref = self.reference_params[name].to(param.device)
            fisher = self.fisher_diagonal[name].to(param.device)
            penalty_loss = penalty_loss + (fisher * (param - param_ref).pow(2)).sum()

        return (self.lambda_ / 2.0) * penalty_loss
