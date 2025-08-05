import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

class PPOAgent:
    def __init__(self, network, optimizer, gamma=0.99, clip=0.2, epochs=4, 
                 entropy_coef=0.01, lam=0.95, batch_size=2048, 
                 max_grad_norm=0.5, target_kl=0.01, lr_schedule=True):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.lam = lam
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.lr_schedule = lr_schedule
        
        # State normalization
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
        
        # Diagnostics
        self.kl_history = deque(maxlen=100)
        self.clip_frac_history = deque(maxlen=100)

    def normalize_states(self, states):
        """Online state normalization"""
        if self.state_mean is None:
            self.state_mean = states.mean(0, keepdim=True)
            self.state_std = states.std(0, keepdim=True) + 1e-8
            self.state_count = 1
        else:
            batch_mean = states.mean(0, keepdim=True)
            batch_std = states.std(0, keepdim=True)
            batch_count = states.shape[0]
            
            # Update running stats
            total_count = self.state_count + batch_count
            delta = batch_mean - self.state_mean
            new_mean = self.state_mean + delta * batch_count / total_count
            
            m_a = self.state_std.pow(2) * (self.state_count - 1)
            m_b = batch_std.pow(2) * (batch_count - 1)
            M2 = m_a + m_b + delta.pow(2) * self.state_count * batch_count / total_count
            new_std = torch.sqrt(M2 / (total_count - 1)) + 1e-8
            
            self.state_mean = new_mean
            self.state_std = new_std
            self.state_count = total_count
            
        return (states - self.state_mean) / self.state_std

    def update(self, states, actions, rewards, log_probs_old, dones, values_old):
        device = next(self.network.parameters()).device
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        values_old = torch.tensor(values_old, dtype=torch.float32, device=device)
        
        # Normalize states
        states = self.normalize_states(states)
        
        # --- GAE + Returns ---
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = values_old[-1].item()
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_advantage = 0
            else:
                next_value = values_old[t+1] if t < len(rewards)-1 else last_value
                
            delta = rewards[t] + self.gamma * next_value - values_old[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values_old
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # --- Mini-batch PPO updates ---
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fracs = []
        
        total_steps = states.shape[0]
        indices = torch.randperm(total_steps, device=device)
        
        # Early stopping
        early_stop = False
        
        for epoch in range(self.epochs):
            if early_stop:
                break
                
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                mb_s = states[idx]
                mb_a = actions[idx]
                mb_lp = log_probs_old[idx]
                mb_ret = returns[idx]
                mb_adv = advantages[idx]
                mb_val_old = values_old[idx]
                
                # Forward pass
                probs, values_pred = self.network(mb_s)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(mb_a)
                entropy = dist.entropy().mean()
                
                # KL divergence
                kl = (mb_lp - log_probs).mean().item()
                kl_divs.append(kl)
                
                # Early stopping if KL too high
                if kl > 1.5 * self.target_kl:
                    early_stop = True
                    break
                
                # Policy loss with clipping
                ratios = torch.exp(log_probs - mb_lp)
                clipped_ratios = torch.clamp(ratios, 1-self.clip, 1+self.clip)
                
                # Clip fraction tracking
                clip_frac = ((ratios > 1+self.clip) | (ratios < 1-self.clip)).float().mean().item()
                clip_fracs.append(clip_frac)
                
                policy_loss = -torch.min(ratios * mb_adv, clipped_ratios * mb_adv).mean()
                
                # Value loss with clipping
                values_pred = values_pred.squeeze(-1)
                values_clipped = mb_val_old + torch.clamp(values_pred - mb_val_old, -self.clip, self.clip)
                v_loss_unclipped = (values_pred - mb_ret).pow(2)
                v_loss_clipped = (values_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
        
        # Update entropy coefficient
        if np.mean(clip_fracs) > 0.3:  # Too many clips → increase exploration
            self.entropy_coef = min(0.05, self.entropy_coef * 1.1)
        elif np.mean(clip_fracs) < 0.1:  # Few clips → decrease exploration
            self.entropy_coef = max(0.001, self.entropy_coef * 0.9)
            
        # Learning rate scheduling
        if self.lr_schedule and np.mean(kl_divs) > self.target_kl:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.8
                
        # Return diagnostics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fracs),
            'entropy_coef': self.entropy_coef
        }
