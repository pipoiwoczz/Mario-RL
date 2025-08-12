import numpy as np
import torch
import torch.nn as nn
from envs.mario_wrappers import MultiMarioEnv
from agent.network import CNNPolicy
from config import FRAME_STACK
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def evaluate_policy(model, env_cls, env_kwargs, n_eval_episodes=10, deterministic=True, render=False):
    """
    Evaluate a trained CNNPolicy in MultiMarioEnv.

    Args:
        model: trained CNNPolicy (torch.nn.Module)
        env_cls: environment class (e.g. MultiMarioEnv)
        env_kwargs: dict of args for creating the environment
        n_eval_episodes: number of episodes to run
        deterministic: if True, select actions greedily
        render: if True, calls env.render() each step
    """
    env = env_cls(num_envs=1, **env_kwargs)
    obs = env.reset()
    episode_rewards = []
    ep_reward = 0
    episodes_done = 0

    model.eval()
    with torch.no_grad():
        while episodes_done < n_eval_episodes:
            # Convert obs to tensor
            obs_t = torch.from_numpy(obs).float() / 255.0
            obs_t = obs_t.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Forward pass
            logits, _ = model(obs_t)
            if deterministic:
                actions = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

            obs, rewards, dones, infos = env.step(actions)
            ep_reward += rewards[0]

            if render:
                env.render()

            if dones[0]:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                episodes_done += 1
                obs = env.reset()

    env.close()
    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    print(f"Evaluation over {n_eval_episodes} episodes: mean reward {mean_r:.2f} ± {std_r:.2f}")
    return mean_r, std_r

if __name__ == "__main__":
    # Load your trained model
    model = CNNPolicy(in_channels=FRAME_STACK, num_actions=len(SIMPLE_MOVEMENT))
    model.load_state_dict(torch.load("trained_model/model_1.1.pth", map_location="cpu"))    # change with your DEVICE (cuda or cpu)

    # Match environment settings from training
    eval_kwargs = {
        "world": 1,
        "stage": 1,
        "action_type": "simple",
        "frame_size": (84, 84),
        "frame_stack": 4,
        "frameskip": 4,
        "output_path": "./eval_videos"
    }

    # Run evaluation
    evaluate_policy(model, MultiMarioEnv, eval_kwargs, n_eval_episodes=5, deterministic=True)
