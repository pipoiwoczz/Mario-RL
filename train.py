import os
import math
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from envs.mario_wrappers import create_train_env
from agent.network import MarioNet
from agent.agent import PPOAgent
import config

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

def main():
    # 1) Environment & device
    env = create_train_env(config.WORLD, config.STAGE, config.ACTION_TYPE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 2) Network, optimizer, agent
    network = MarioNet(config.INPUT_DIMS, env.action_space.n, device=device).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.LR)
    agent = PPOAgent(
        network=network,
        optimizer=optimizer,
        gamma=config.GAMMA,
        clip=config.CLIP,
        epochs=config.EPOCHS,
        lam=0.95,
        batch_size=config.BATCH_SIZE
    )

    # 3) Training loop
    for episode in range(1, config.EPISODES + 1):
        state, _ = env.reset()
        state = np.squeeze(state, axis=0)  # [1,4,84,84] → [4,84,84]
        
        episode_reward = 0
        done = False
        step_count = 0

        # PPO buffers
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []

        pbar = tqdm(total=config.MAX_STEPS, desc=f"Episode {episode}")

        while not done and step_count < config.MAX_STEPS:
            # normalize & to-tensor
            state_tensor = (
                torch.tensor(state / 255.0, dtype=torch.float32)
                     .unsqueeze(0)
                     .to(device)
            )

            # 1) rollout: get action, log_prob, and value
            with torch.no_grad():
                probs, value = network(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # step in env
            next_state, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc
            next_state = np.squeeze(next_state, axis=0)

            # store
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)
            values.append(value.item())

            # advance
            state = next_state
            episode_reward += reward
            step_count += 1
            pbar.update(1)
            pbar.set_postfix(reward=episode_reward)

            # 2) update whenever we hit batch size
            if step_count % config.BATCH_SIZE == 0:
                agent.update(
                    states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    log_probs_old=np.array(log_probs),
                    dones=np.array(dones),
                    values=np.array(values)
                )
                # clear buffers
                states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []

        pbar.close()

        # final update on leftovers
        if states:
            agent.update(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                log_probs_old=np.array(log_probs),
                dones=np.array(dones),
                values=np.array(values)
            )

        # checkpoints
        if episode % config.SAVE_INTERVAL == 0:
            path = f"checkpoints/mario_ppo_{episode}.pth"
            torch.save(network.state_dict(), path)
            print(f"Saved checkpoint: {path}")

        print(f"Episode {episode} → reward: {episode_reward:.2f}, steps: {step_count}")

    # final save
    torch.save(network.state_dict(), "checkpoints/mario_ppo_final.pth")
    env.close()

if __name__ == "__main__":
    main()
