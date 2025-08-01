import  gym
import cv2
import numpy as np
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from env.monitor import Monitor
import gym_super_mario_bros

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame
    return np.zeros((1, 84, 84))

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, world=None, stage=None, monitor=None):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        self.monitor = monitor

    def step(self, action):
        # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if self.monitor:
            self.monitor.record(state)
            
        state = process_frame(state)
        done = terminated or truncated
        
        # Update reward based on score
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]
        
        # Add bonus for completing level
        if done:
            reward += 50 if info["flag_get"] else -50
            
        # World-specific death conditions
        if self.world == 7 and self.stage == 4:
            if self._is_death_condition_w7s4(info):
                reward -= 50
                terminated = True
                
        if self.world == 4 and self.stage == 4:
            if self._is_death_condition_w4s4(info):
                reward = -50
                terminated = True
                
        self.current_x = info["x_pos"]
        return state, reward / 10.0, terminated, truncated, info

    def reset(self, **kwargs):
        self.curr_score = 0
        self.current_x = 40
        # Gymnasium reset returns 2 values: obs, info
        obs, info = self.env.reset(**kwargs)
        return process_frame(obs), info
    
    def _is_death_condition_w7s4(self, info):
        x, y = info["x_pos"], info["y_pos"]
        return (
            (506 <= x <= 832 and y > 127) or
            (832 < x <= 1064 and y < 80) or
            (1113 < x <= 1464 and y < 191) or
            (1579 < x <= 1943 and y < 191) or
            (1946 < x <= 1964 and y >= 191) or
            (1984 < x <= 2060 and (y >= 191 or y < 127)) or
            (2114 < x < 2440 and y < 191) or
            x < self.current_x - 500
        )
    
    def _is_death_condition_w4s4(self, info):
        x, y = info["x_pos"], info["y_pos"]
        return (
            (x <= 1500 and y < 127) or
            (1588 <= x < 2380 and y >= 127)
        )

class CustomSkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        terminated = False
        truncated = False
        info = {}
        
        for i in range(self.skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if i >= self.skip / 2:
                last_states.append(state)
                
            if terminated or truncated:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, terminated, truncated, info
        
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        
        return self.states[None, :, :, :].astype(np.float32), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32), info

def create_train_env(world, stage, action_type="simple", output_path=None):
    try:
        # Try modern environment ID first
        env = gym_super_mario_bros.make(
            f"SuperMarioBros-{world}-{stage}-v3",
            apply_api_compatibility=True,
            render_mode="human"
        )
    except gym.error.Error:
        # Fallback to v0 environment
        env = gym.make(
            "SuperMarioBros-v0",
            apply_api_compatibility=True,
            render_mode="human"
        )
    
    monitor = Monitor(256, 240, output_path) if output_path else None
    
    action_map = {
        "right": RIGHT_ONLY,
        "simple": SIMPLE_MOVEMENT,
        "complex": COMPLEX_MOVEMENT
    }
    actions = action_map.get(action_type, SIMPLE_MOVEMENT)
    
    env = JoypadSpace(env, actions)
    env = CustomRewardWrapper(env, world, stage, monitor)
    env = CustomSkipFrameWrapper(env)
    
    return env