import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, network, optimizer, gamma=0.99, clip=0.2, epochs=4, entropy_coef=0.01, lam=0.95, batch_size=128):
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.lam = lam
        self.batch_size = batch_size

    def update(self, states=None, actions=None, rewards=None, 
           log_probs_old=None, dones=None, values=None):

        device = next(self.network.parameters()).device

        # to‐tensor, device
        states       = torch.tensor(states,      dtype=torch.float32, device=device)
        actions      = torch.tensor(actions,     dtype=torch.long,   device=device)
        rewards      = torch.tensor(rewards,     dtype=torch.float32, device=device)
        dones        = torch.tensor(dones,       dtype=torch.float32, device=device)
        log_probs_old= torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        values       = torch.tensor(values,      dtype=torch.float32, device=device)

        # --- 1) GAE + Returns ---
        advantages = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns    = advantages + values

        # --- 2) Protect against singleton normalization ---
        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std  = advantages.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # only one sample → zero-advantage
            advantages = torch.zeros_like(advantages)

        # --- 3) Mini-batch PPO updates ---
        total_steps = states.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, self.batch_size):
                idx = perm[start : start + self.batch_size]

                mb_s = states[idx]
                mb_a = actions[idx]
                mb_lp= log_probs_old[idx]
                mb_ret = returns[idx]
                mb_adv = advantages[idx]

                # forward
                probs, values_pred = self.network(mb_s)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(mb_a)
                entropy   = dist.entropy().mean()

                # PPO losses
                ratios = torch.exp(log_probs - mb_lp)
                surr1  = ratios * mb_adv
                surr2  = torch.clamp(ratios, 1-self.clip, 1+self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # use squeeze(-1) to only drop the last dim, preserving batch dim
                value_loss  = F.mse_loss(values_pred.squeeze(-1), mb_ret)

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

            # logging (uses last batch’s stats)
            # with torch.no_grad():
            #     approx_kl = (mb_lp - log_probs).mean().item()
            #     clip_frac = ((ratios > 1+self.clip) | (ratios < 1-self.clip)).float().mean().item()
                # print(f"Epoch KL={approx_kl:.4f}, clip_frac={clip_frac:.4f}, "
                #     f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}")
