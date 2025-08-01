import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, network, optimizer, gamma=0.99, clip=0.2, epochs=4, entropy_coef=0.01):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef  # Make configurable

    def update(self, states=None, actions=None, rewards=None, 
               log_probs_old=None, dones=None):
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float)

        # Calculate discounted returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.epochs):
            # Calculate new probabilities and values
            probs, values = self.network(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate policy loss
            ratios = torch.exp(log_probs - log_probs_old.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss with increased entropy coefficient
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()