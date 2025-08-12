import os
import imageio
import numpy as np
import multiprocessing as mp
import cv2
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from mario_wrappers import  MultiMarioEnv

if __name__ == "__main__":
    # Test with safe parameters
    env = MultiMarioEnv(
        num_envs=2,
        action_type="right",
        frameskip=4,  # Ensure valid value
        frame_size=(84, 84),
        frame_stack=4,
        output_path="./test"
    )

    try:
        for _ in range(3):  # Multiple reset attempts
            try:
                obs = env.reset()
                print("Obs shape:", obs.shape)
                break
            except Exception as e:
                print(f"Reset failed: {e}")
                env.close()
                env = MultiMarioEnv(num_envs=2, action_type="right", frameskip=4)

        for i in range(10):
            actions = np.random.randint(0, len(env.actions), size=(env.num_envs,))
            obs, rewards, dones, infos = env.step(actions)
            print(f"Step {i}: Rewards={rewards}, Dones={dones}")

            if any(dones):
                print("Resetting finished environments")
                env.reset()

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        env.close()
        print("Test completed")