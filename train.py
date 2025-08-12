from envs.mario_wrappers import MultiMarioEnv
import numpy as np
from agent.network import CNNPolicy
from agent.agent import PPOTrainer
from config import NUM_ENVS, FRAME_STACK, ROLLOUT_LEN, MINIBATCHES, PPO_EPOCHS, LR

if __name__ == "__main__":
    eval_kwargs = {
        "world": 1,
        "stage": 1,
        "action_type": "simple",
        "frame_size": (84, 84),
        "frame_stack": 4,
        "frameskip": 4,
        "output_path": "./training_videos"  # Enable video recording to show the training process
    }

    env = MultiMarioEnv(
        num_envs=NUM_ENVS,
        action_type=eval_kwargs["action_type"],
        frame_size=eval_kwargs["frame_size"],
        frame_stack=eval_kwargs["frame_stack"],
        frameskip=eval_kwargs["frameskip"],
        output_path=eval_kwargs["output_path"]
    )

    NUM_ACTIONS = len(env.actions)
    model = CNNPolicy(in_channels=FRAME_STACK, num_actions=NUM_ACTIONS)

    trainer = PPOTrainer(
        env,
        model,
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        minibatches=MINIBATCHES,
        ppo_epochs=PPO_EPOCHS,
        lr=LR,
        eval_env_kwargs=eval_kwargs
    )

    trainer.train(
        total_updates=1000,
        save_every=30,
        log_interval=1,
        evaluate_every=10,
        n_eval_episodes=20,
        deterministic_eval=True
    )