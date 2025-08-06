import os
import torch
import numpy as np
from tqdm import tqdm
import config
from envs.mario_wrappers import create_train_env
from agent.agent import PPOAgent
from agent.network import MarioNet
from utils import setup_logging, save_checkpoint, load_checkpoint, log_episode

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logging(config.DRIVE_DIR)
    
    # Create environment and model
    env = create_train_env(config.WORLD, config.STAGE, config.ACTION_TYPE)
    model = MarioNet(config.INPUT_DIMS, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    # Load checkpoint if exists
    start_episode = load_checkpoint(model, optimizer, config.DRIVE_DIR, config.LOAD_EPISODE)
    
    # Initialize agent
    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        gamma=config.GAMMA,
        clip=config.CLIP,
        entropy_coef=config.ENTROPY_COEF
    )
    
    # Training loop
    for episode in range(start_episode, config.EPISODES + 1):
        state, _ = env.reset()
        state = np.squeeze(state, axis=0)  # Remove batch dimension
        
        episode_reward = 0
        done = False
        step_count = 0
        max_x = 0  # Track progress
        
        # Experience buffers
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        
        # Progress bar
        pbar = tqdm(total=config.MAX_STEPS, desc=f"Episode {episode}")
        
        while not done and step_count < config.MAX_STEPS:
            # Prepare state tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            
            # Get action and value
            with torch.no_grad():
                probs, value = model(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Environment step
            next_state, reward, term, trunc, info = env.step(action.item())
            done = term or trunc
            next_state = np.squeeze(next_state, axis=0)
            
            # Track progress
            current_x = info.get("x_pos", 0)
            if current_x > max_x:
                max_x = current_x
            
            # Store experience
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)
            values.append(value.item())
            
            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1
            pbar.update(1)
            pbar.set_postfix(reward=episode_reward, max_x=max_x)
            
            # Update agent if batch is complete
            if step_count % config.BATCH_SIZE == 0:
                agent.update(
                    states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    log_probs_old=np.array(log_probs),
                    dones=np.array(dones),
                    values_old=np.array(values)
                )
                # Reset buffers
                states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        
        pbar.close()
        
        # Final update with remaining experiences
        if states:
            agent.update(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                log_probs_old=np.array(log_probs),
                dones=np.array(dones),
                values_old=np.array(values)
            )
        
        # Log episode stats
        log_episode(
            episode, 
            episode_reward, 
            step_count, 
            max_x, 
            agent.stats if hasattr(agent, 'stats') else {}
        )
        
        # Save checkpoint periodically
        if episode % config.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, config.DRIVE_DIR, episode)
    
    # Final save
    save_checkpoint(model, optimizer, config.DRIVE_DIR, "final")
    env.close()

if __name__ == "__main__":
    main()