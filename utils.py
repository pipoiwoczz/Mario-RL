import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import config

def setup_logging(log_dir):
    """Create directories for logging"""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    
def load_checkpoint(model, optimizer, log_dir, episode=0):
    """Load model checkpoint if exists"""
    start_episode = 0
    if episode > 0:
        checkpoint_path = os.path.join(log_dir, "checkpoints", f"mario_ppo_{episode}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            print(f"Loaded checkpoint from episode {episode}")
    return start_episode

def save_checkpoint(model, optimizer, log_dir, episode):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(log_dir, "checkpoints", f"mario_ppo_{episode}.pth")
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint at episode {episode}")

def log_episode(episode, reward, steps, max_x, stats):
    """Log episode statistics and update progress plot"""
    # Append to CSV log
    log_path = os.path.join(config.DRIVE_DIR, "training_log.csv")
    log_data = {
        'episode': episode,
        'reward': reward,
        'steps': steps,
        'max_x': max_x,
        'policy_loss': stats.get('policy_loss', 0),
        'value_loss': stats.get('value_loss', 0),
        'entropy': stats.get('entropy', 0),
        'timestamp': datetime.now().isoformat()
    }
    
    df = pd.DataFrame([log_data])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
    
    # Update progress plot every 10 episodes
    if episode % 10 == 0:
        plot_progress(log_path)

def plot_progress(log_path):
    """Generate training progress plot"""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        
        plt.figure(figsize=(15, 10))
        
        # Reward progression
        plt.subplot(2, 2, 1)
        plt.plot(df['episode'], df['reward'], 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Progression')
        plt.grid(True)
        
        # Max X position
        plt.subplot(2, 2, 2)
        plt.plot(df['episode'], df['max_x'], 'g-')
        plt.xlabel('Episode')
        plt.ylabel('Max X Position')
        plt.title('Level Progress')
        plt.grid(True)
        
        # Losses
        plt.subplot(2, 2, 3)
        if 'policy_loss' in df and 'value_loss' in df:
            plt.plot(df['episode'], df['policy_loss'], 'r-', label='Policy Loss')
            plt.plot(df['episode'], df['value_loss'], 'c-', label='Value Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
        
        # Entropy
        plt.subplot(2, 2, 4)
        if 'entropy' in df:
            plt.plot(df['episode'], df['entropy'], 'm-')
            plt.xlabel('Episode')
            plt.ylabel('Entropy')
            plt.title('Exploration')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.DRIVE_DIR, 'training_progress.png'))
        plt.close()