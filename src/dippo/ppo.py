import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        h = self.shared(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs, mask=None):
        logits, value = self(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, obs, actions, mask=None):
        logits, value = self(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()


def ppo_update(model, optimizer, batch, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4):
    obs = batch['obs']
    actions = batch['actions']
    old_logp = batch['logp'].detach()
    returns = batch['returns'].detach()
    adv = batch['adv'].detach()
    masks = batch['masks']
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    last_loss = None
    for _ in range(epochs):
        logp, value, entropy = model.evaluate_actions(obs, actions, masks)
        ratio = torch.exp(logp - old_logp)
        pg1 = ratio * adv
        pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -torch.min(pg1, pg2).mean()
        value_loss = F.mse_loss(value, returns)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()
    return last_loss
