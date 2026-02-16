import torch


class Deterministic_EWC:
    """
    Diagonal EWC regularizer for deterministic actor-critic agents (e.g., DDPG).

    Fisher importance is approximated from actor gradients of the deterministic
    policy objective: maximize Q(s, pi(s)), i.e. minimize -Q(s, pi(s)).
    """

    def __init__(self, agent, dataloader, lambda_=0.4):
        self.lambda_ = lambda_
        self.device = agent.device
        self.reference_params = {
            name: param.detach().clone()
            for name, param in agent.actor.named_parameters()
            if param.requires_grad
        }
        self.fisher_diagonal = self._estimate_fisher(agent, dataloader)

    def _estimate_fisher(self, agent, dataloader):
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.reference_params.items()
        }

        # Freeze critic parameters while estimating actor importance.
        critic_params = list(agent.critic.parameters())
        critic_requires_grad = [param.requires_grad for param in critic_params]
        for param in critic_params:
            param.requires_grad_(False)

        num_batches = 0
        try:
            for data in dataloader:
                obs = data[0].to(self.device)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                actions = agent.actor(obs)
                actor_objective = -agent.critic.q1_forward(obs, actions).mean()

                agent.actor.optimizer.zero_grad()
                actor_objective.backward()

                for name, param in agent.actor.named_parameters():
                    if name in fisher and param.grad is not None:
                        fisher[name] += param.grad.detach().pow(2)

                num_batches += 1
        finally:
            for param, requires_grad in zip(critic_params, critic_requires_grad):
                param.requires_grad_(requires_grad)
            agent.actor.optimizer.zero_grad()

        if num_batches == 0:
            return fisher

        for name in fisher:
            fisher[name] /= num_batches

        return fisher

    def penalty(self, target_actor):
        penalty_loss = torch.zeros(1, device=next(target_actor.parameters()).device)

        for name, param in target_actor.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.reference_params:
                continue

            reference = self.reference_params[name].to(param.device)
            fisher = self.fisher_diagonal[name].to(param.device)
            penalty_loss = penalty_loss + (fisher * (param - reference).pow(2)).sum()

        return (self.lambda_ / 2.0) * penalty_loss
