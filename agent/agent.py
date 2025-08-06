import torch
import torch.nn.functional as F
import numpy as np

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99, clip=0.2, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.stats = {}  # For tracking training metrics
        
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        next_value = 0
        
        # Traverse backwards through timesteps
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * 0.95 * gae
            
            advantages.insert(0, gae)
            next_value = values[t]
        
        return torch.tensor(advantages, dtype=torch.float32, device=values.device)

    def update(self, states, actions, rewards, log_probs_old, dones, values_old):
        device = next(self.model.parameters()).device
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=device) / 255.0
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        values_old = torch.tensor(values_old, dtype=torch.float32, device=device)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values_old, dones)
        returns = advantages + values_old
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy and values
        probs, values = self.model(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate policy loss (clipped surrogate objective)
        ratios = torch.exp(log_probs - log_probs_old.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Store stats for logging
        self.stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }