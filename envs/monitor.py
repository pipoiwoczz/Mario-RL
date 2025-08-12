import os
import imageio
import numpy as np

class Monitor:
    def __init__(self, env, video_folder, record_every=1):
        self.env = env
        self.video_folder = video_folder
        self.record_every = record_every
        self.episode_id = 0
        self.frames = []
        self.highest_reward = -float('inf')
        os.makedirs(video_folder, exist_ok=True)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.frames = []
        self.current_reward = 0
        if self.record_every > 0 and self.episode_id % self.record_every == 0:
            frame = self._get_frame()
            self.frames.append(frame)
        return result

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_reward += reward
        
        if self.record_every > 0 and self.episode_id % self.record_every == 0:
            frame = self._get_frame()
            self.frames.append(frame)
            
        if done and self.current_reward > self.highest_reward:
            for _ in range(15):  # adjust number to match animation length
                frame = self._get_frame()
                self.frames.append(frame)
            if self.current_reward > self.highest_reward:
                if info.get('flag_get'):
                    print("Level completed")
                self.highest_reward = self.current_reward
                self._save_video()
        
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def _get_frame(self):
        frame = self.env.render(mode="rgb_array")
        return np.array(frame)

    def _save_video(self):
        if not self.frames:
            return
            
        path = os.path.join(self.video_folder, f"highscore_{self.episode_id}_{int(self.highest_reward)}.mp4")
        imageio.mimsave(path, self.frames, fps=30)
        print(f"[Monitor] New high score! Saved video: {path}")

    def end_episode(self):
        if self.episode_id % self.record_every == 0 and self.frames:
            path = os.path.join(self.video_folder, f"episode_{self.episode_id}.mp4")
            imageio.mimsave(path, self.frames, fps=30)
        
        self.episode_id += 1
