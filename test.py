import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from envs.mario_wrappers import create_train_env
import warnings
from gym.spaces import Discrete
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)

env = create_train_env(1, 1, 'complex', 'out', version='v0')

obs, info = env.reset()
done = False

        
while not done:
    action = env.action_space.sample()  # or your agent’s action
    # print all action_space and stop
    if isinstance(env.action_space, Discrete):
        for act in range(env.action_space.n):
            print(act)
        sys.pause()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()