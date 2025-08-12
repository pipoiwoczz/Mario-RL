import torch
import numpy as _np
import os
from torch.distributions import Categorical
import time
import torch.nn.functional as F
import torch.nn as nn


from config import SAVE_DIR, LOG_PATH, DEVICE, GAMMA, GAE_LAMBDA, CLIP_EPS, LR, ENT_COEF, VAL_COEF, MAX_GRAD_NORM, ADV_SCALE, NUM_ENVS \
    , FRAME_STACK, ROLLOUT_LEN, MINIBATCHES, PPO_EPOCHS, C, H, W
from envs.mario_wrappers import MultiMarioEnv

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def obs_to_tensor(obs_np):
    """(N,H,W,C) uint8 -> (N,C,H,W) float32 on DEVICE in [0,1]."""
    arr = _np.asarray(obs_np, dtype=_np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr).to(DEVICE)


def compute_gae_torch(rewards, values, dones, last_values, gamma=GAMMA, lam=GAE_LAMBDA):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards, device=DEVICE)
    last_adv = torch.zeros(N, device=DEVICE)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_value = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def explained_variance(y_pred, y):
    y = y.detach()
    y_pred = y_pred.detach()
    var_y = torch.var(y)
    return float(1 - torch.var(y - y_pred) / (var_y + 1e-8))


# Main Training Loop
class PPOTrainer:
    def __init__(self, env, model, num_envs=NUM_ENVS, rollout_len=ROLLOUT_LEN,
                 minibatches=MINIBATCHES, ppo_epochs=PPO_EPOCHS, lr=LR,
                 eval_env_kwargs=None):
        # Initialize environment and model
        self.env = env
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.minibatches = minibatches
        self.ppo_epochs = ppo_epochs
        self.batch_size = num_envs * rollout_len
        self.minibatch_size = self.batch_size // minibatches

        self.global_step = 0
        self.update_count = 0
        self.log_path = LOG_PATH

        self.best_mean_eval = -float("inf")
        self.eval_env_kwargs = eval_env_kwargs or {}
        
        # Add tracking for high scores per environment
        self.env_params = {
            'world': eval_env_kwargs.get('world', 1),
            'stage': eval_env_kwargs.get('stage', 1),
            'action_type': eval_env_kwargs.get('action_type', 'simple'),
            'frame_size': eval_env_kwargs.get('frame_size', (84, 84)),
            'frame_stack': eval_env_kwargs.get('frame_stack', 4),
            'frameskip': eval_env_kwargs.get('frameskip', 4),
            'output_path': eval_env_kwargs.get('output_path', None)
        }
        self.high_scores = [-float('inf')] * num_envs
        self.worker_restart_count = 0

    def _recreate_env(self):
        # Safely recreate the environment
        print("Recreating environment due to worker failure...")
        try:
            self.env.close()
        except:
            pass
            
        new_env = MultiMarioEnv(
            num_envs=self.num_envs,
            action_type=self.env_params['action_type'],
            frameskip=self.env_params['frameskip'],
            frame_size=self.env_params['frame_size'],
            frame_stack=self.env_params['frame_stack'],
            output_path=self.env_params['output_path']
        )
        
        self.worker_restart_count += 1
        self.high_scores = [-float('inf')] * self.num_envs
        return new_env

    def collect_rollouts(self):
        T = self.rollout_len
        N = self.num_envs

        log_interval = 10
        self.ep_rewards = getattr(self, "ep_rewards", [])
        self.running_rewards = getattr(self, "running_rewards", [0] * N)

        obs_buf = torch.zeros(T, N, C, H, W, dtype=torch.float32, device=DEVICE)
        actions_buf = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
        logp_buf = torch.zeros(T, N, dtype=torch.float32, device=DEVICE)
        rewards_buf = torch.zeros(T, N, dtype=torch.float32, device=DEVICE)
        dones_buf = torch.zeros(T, N, dtype=torch.float32, device=DEVICE)
        values_buf = torch.zeros(T, N, dtype=torch.float32, device=DEVICE)

        # Attempt rollout with recovery
        for attempt in range(3):  
            try:
                obs_np = self.env.reset()
                obs_t = obs_to_tensor(obs_np)
                break
            except (ConnectionResetError, BrokenPipeError):
                if attempt < 2:
                    self.env = self._recreate_env()
                else:
                    raise RuntimeError("Failed to reset environment after 3 attempts")
        
        for t in range(T):
            try:
                with torch.no_grad():
                    logits, values = self.model(obs_t)
                    dist = Categorical(logits=logits)
                    actions_t = dist.sample()
                    logps_t = dist.log_prob(actions_t)

                actions_np = actions_t.cpu().numpy().astype(_np.int32)
                next_obs_np, rewards_np, dones_np, infos = self.env.step(actions_np)

                # Check for new high scores and trigger video recording
                for i in range(N):
                    self.running_rewards[i] += rewards_np[i]
                    if dones_np[i]:
                        self.ep_rewards.append(self.running_rewards[i])
                        self.running_rewards[i] = 0
                        if len(self.ep_rewards) % log_interval == 0:
                            avg_r = _np.mean(self.ep_rewards[-log_interval:])
                            print(f"[Train] Episodes {len(self.ep_rewards)} - Avg reward (last {log_interval} eps): {avg_r:.2f}")

                obs_buf[t].copy_(obs_t)
                actions_buf[t].copy_(actions_t)
                logp_buf[t].copy_(logps_t)
                rewards_buf[t].copy_(torch.from_numpy(_np.asarray(rewards_np, dtype=_np.float32)).to(DEVICE))
                dones_buf[t].copy_(torch.from_numpy(_np.asarray(dones_np, dtype=_np.float32)).to(DEVICE))
                values_buf[t].copy_(values)

                obs_np = next_obs_np
                obs_t = obs_to_tensor(obs_np)
                self.global_step += N
                
            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"Step {t} failed: {e}")
                self.env = self._recreate_env()
                
                if t > T // 2:
                    return self.collect_rollouts()
                else:
                    print(f"Continuing with partial rollout ({t}/{T} steps)")
                    break

        with torch.no_grad():
            _, last_values = self.model(obs_t)
        last_values = last_values.to(DEVICE)

        # Compute GAE
        actual_steps = t + 1 if t < T else T
        advantages, returns = compute_gae_torch(
            rewards_buf[:actual_steps], 
            values_buf[:actual_steps], 
            dones_buf[:actual_steps], 
            last_values
        )

        return {
            "obs": obs_buf[:actual_steps],
            "actions": actions_buf[:actual_steps],
            "logprobs": logp_buf[:actual_steps],
            "rewards": rewards_buf[:actual_steps],
            "dones": dones_buf[:actual_steps],
            "values": values_buf[:actual_steps],
            "advantages": advantages,
            "returns": returns
        }

    def update(self, batch):
        T, N = batch["actions"].shape
        B = T * N

        b_obs = batch["obs"].reshape(B, C, H, W)
        b_actions = batch["actions"].reshape(B).to(DEVICE)
        b_old_logp = batch["logprobs"].reshape(B).to(DEVICE)
        b_adv = batch["advantages"].reshape(B).to(DEVICE)
        b_returns = batch["returns"].reshape(B).to(DEVICE)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8) * ADV_SCALE
        inds = _np.arange(B)

        tot_policy_loss = 0.0
        tot_value_loss = 0.0
        tot_entropy = 0.0
        tot_kl = 0.0
        tot_clip = 0.0
        mb_count = 0

        for epoch in range(self.ppo_epochs):
            _np.random.shuffle(inds)
            for start in range(0, B, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]
                mb_inds_t = torch.from_numpy(mb_inds).long().to(DEVICE)

                mb_obs = b_obs.index_select(0, mb_inds_t)
                mb_actions = b_actions.index_select(0, mb_inds_t)
                mb_old_logp = b_old_logp.index_select(0, mb_inds_t)
                mb_adv = b_adv.index_select(0, mb_inds_t)
                mb_returns = b_returns.index_select(0, mb_inds_t)

                logits, values = self.model(mb_obs)
                dist = Categorical(logits=logits)
                mb_logp = dist.log_prob(mb_actions)
                mb_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                loss = policy_loss + VAL_COEF * value_loss - ENT_COEF * mb_entropy

                with torch.no_grad():
                    approx_kl = (mb_old_logp - mb_logp).mean().item()
                    # early stopping for kl
                    if approx_kl >  0.03:
                        break
                    clip_frac = (((ratio > 1.0 + CLIP_EPS) | (ratio < 1.0 - CLIP_EPS)).float().mean().item())
                    ev = explained_variance(values.squeeze(-1), mb_returns)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                tot_policy_loss += float(policy_loss.item())
                tot_value_loss += float(value_loss.item())
                tot_entropy += float(mb_entropy.item())
                tot_kl += float(approx_kl)
                tot_clip += float(clip_frac)
                mb_count += 1

        if mb_count > 0:
            avg_policy_loss = tot_policy_loss / mb_count
            avg_value_loss = tot_value_loss / mb_count
            avg_entropy = tot_entropy / mb_count
            avg_kl = tot_kl / mb_count
            avg_clip = tot_clip / mb_count
            print(f"[PPO] Update {self.update_count+1} epochs={self.ppo_epochs} policy_loss={avg_policy_loss:.4f} value_loss={avg_value_loss:.4f} entropy={avg_entropy:.4f} clip_frac={avg_clip:.4f} approx_kl={avg_kl:.6f}")

        self.update_count += 1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, total_updates=10000, save_every=50, log_interval=1):
        t0 = time.time()
        for update in range(1, total_updates + 1):
            batch = self.collect_rollouts()
            self.update(batch)

            if update % save_every == 0:
                self.save(os.path.join(SAVE_DIR, f"ppo_{update}.pth"))

            if update % log_interval == 0:
                elapsed = time.time() - t0
                print(f"[Update {update}]: global_steps={self.global_step} elapsed={elapsed:.1f}s")

        print("Training finished.")