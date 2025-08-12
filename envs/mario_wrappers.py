import os
import imageio
import numpy as np
import multiprocessing as mp
import cv2
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from envs.monitor import Monitor

def preprocess_frame(frame, size=(84, 84)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def make_env(world, stage, actions, frame_size=(84, 84), frame_stack=4, frameskip=4, output_path=None):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v3")
    env = JoypadSpace(env, actions)
    env = Monitor(env, output_path, record_every=1) if output_path else env
    env.frame_size = frame_size
    env.frame_stack = frame_stack
    env.frameskip = frameskip
    return env

def worker_process(conn, world, stage, actions, frame_size, frame_stack, frameskip, output_path):
    """
    Worker process to interact with the environment and communicate with the main process.
    """
    try:
        actual_frameskip = max(1, int(frameskip))
        env = make_env(world, stage, actions, frame_size, frame_stack, actual_frameskip, output_path)
        state_buffer = deque(maxlen=frame_stack)
        is_monitor = isinstance(env, Monitor)
        frame_stack = frame_stack
        frame_size = frame_size
        
        def reset_env():
            obs = env.reset()
            frame = preprocess_frame(obs, frame_size)
            state_buffer.clear()
            for _ in range(frame_stack):
                state_buffer.append(frame)
            return np.stack(state_buffer, axis=-1)

        prev_info = {"x_pos": 0}
        state = reset_env()

        while True:
            try:
                cmd, data = conn.recv()
            except (EOFError, ConnectionResetError):
                break
                
            if cmd == "reset":
                state = reset_env()
                conn.send(state)
            elif cmd == "step":
                action = data
                total_reward = 0
                done = False
                info = {}
                
                for _ in range(actual_frameskip):
                    if is_monitor:
                        obs, reward, done, info = env.step(action)
                    else:
                        obs, reward, done, info = env.step(action)
                    
                    prev_info.update(info)

                    # Check for new high scores
                    if info['flag_get']:
                        reward += 500
                    if info.get("x_pos") > 2000:
                        reward += 2
                    elif info.get("x_pos") > 3000:
                        reward += 10
                    total_reward += reward
                    if done:
                        break
                
                if not done:
                    frame = preprocess_frame(obs, frame_size)
                    state_buffer.append(frame)
                    state = np.stack(state_buffer, axis=-1)
                else:
                    if is_monitor:
                        env.end_episode()
                    state = reset_env()
                
                conn.send((state, total_reward / 10, done, info))
            elif cmd == "record":
                # don't use yet, just let the Monitor auto-handle it
                if is_monitor:
                    pass
                # fallback
                try:
                    conn.send((state, 0.0, False, {}))
                except Exception:
                    try:
                        conn.send((np.stack([np.zeros(frame_size, dtype=np.uint8)]*frame_stack, axis=-1),
                                   0.0, False, {}))
                    except Exception:
                        conn.send((np.zeros((frame_size[0], frame_size[1], frame_stack), dtype=np.uint8),
                                   0.0, False, {}))
            elif cmd == "close":
                break
    except Exception as e:
        print(f"Worker process error: {e}")
    finally:
        env.close()
        conn.close()

class MultiMarioEnv:
    def __init__(self, world=1, stage=1, action_type="simple",
                 num_envs=2, frame_size=(84, 84), frame_stack=4, frameskip=4, output_path=None):
        
        if action_type == "right":
            self.actions = RIGHT_ONLY
        elif action_type == "simple":
            self.actions = SIMPLE_MOVEMENT
        else:
            self.actions = COMPLEX_MOVEMENT
        
        self.frame_size = frame_size    
        self.frame_stack = frame_stack   

        self.num_envs = num_envs
        self.parent_conns, self.worker_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        self.output_path = output_path

        for idx in range(num_envs):
            p = mp.Process(
                target=worker_process,
                args=(self.worker_conns[idx], world, stage, self.actions,
                      frame_size, frame_stack, frameskip, output_path),
                daemon=True
            )
            p.start()
            self.ps.append(p)

        # Close worker connections in parent process
        for conn in self.worker_conns:
            conn.close()
            
    def trigger_recording(self, env_idx, timeout=5.0):
        # Trigger video recording for a specific environment
        if not self.output_path:
            return False
        conn = self.parent_conns[env_idx]
        # don't use yet, just let the Monitor auto-handle it
        try:
            conn.send(("record", None))
            resp = conn.recv()
            return True
        except Exception as e:
            print(f"Error triggering recording for env {env_idx}: {e}")
            return False

    def reset(self):
      results = []
      for conn in self.parent_conns:
          try:
              conn.send(("reset", None))
          except Exception as e:
              print(f"Send error: {e}")
      for conn in self.parent_conns:
          try:
              results.append(conn.recv())
          except EOFError:
              print("One worker died during reset; returning dummy obs.")
              # use parent env attributes (safe) instead of Process attributes
              h, w = self.frame_size
              fs = self.frame_stack
              results.append(np.zeros((h, w, fs), dtype=np.uint8))
      # debug: verify shapes
      shapes = [r.shape for r in results]
      if len(set(shapes)) != 1:
          print("[MultiMarioEnv.reset] shape mismatch on reset:", shapes)
      return np.stack(results)

    def step(self, actions):
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self.parent_conns]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(dones), infos

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except BrokenPipeError:
                pass
                
        for p in self.ps:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()