# Mario-RL

[![GitHub Repo](https://img.shields.io/badge/GitHub-Mario--RL-black?logo=github)](https://github.com/pipoiwoczz/Mario-RL) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2C3.9-blue)](https://www.python.org/) [![Deps](https://img.shields.io/badge/deps-numpy%20%7C%20gym--super--mario--bros-lightgrey)](https://pypi.org/)

<video width="640" controls>
  <source src="eval_videos/map_1.1.gif" type="video/gif">
  Your browser does not support the video tag.
</video>

Mario-RL is a reinforcement learning project designed to train an agent to navigate and excel in the classic Super Mario Bros game environment using advanced RL algorithms. The project leverages a multiprocessing-ready environment wrapper (`MultiMarioEnv`) and a monitoring system to record high-score videos, making it ideal for RL experimentation. It is currently in active development, with ongoing improvements to training stability, agent performance, and environment robustness.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick Start / Usage Examples](#quick-start--usage-examples)
7. [Monitor & Recording Details](#monitor--recording-details)
8. [Training Hyperparameters](#training-hyperparameters)
9. [Common Problems & Troubleshooting](#common-problems--troubleshooting)
10. [Customization Tips](#customization-tips)
11. [Contributing](#contributing)
12. [Acknowledgements](#acknowledgements)
13. [License & Contact](#license--contact)

---

## Project Overview

Mario-RL provides a lightweight, multiprocessing-enabled environment wrapper for the `gym_super_mario_bros` environment, optimized for reinforcement learning experiments. The core component, `MultiMarioEnv`, runs multiple Mario environments in parallel using separate processes, ensuring robust and isolated emulator execution. The project includes a `Monitor` wrapper that captures and saves video recordings of high-score episodes, facilitating performance evaluation and visualization.

The project focuses on:
- **Parallel Environments**: Utilizes `multiprocessing.Process` and `Pipe` for efficient parallel execution of multiple game instances.
- **Frame Preprocessing**: Converts game frames to grayscale and resizes them (default: 84×84) with frame stacking for temporal context.
- **Video Recording**: Automatically saves `.mp4` videos when an environment achieves a new high score.
- **RL Compatibility**: Designed to integrate seamlessly with popular RL algorithms like PPO, DQN, or A2C.

This project is ideal for researchers, students, and hobbyists interested in applying reinforcement learning to complex, dynamic game environments.

## Features

- **MultiMarioEnv**: Runs multiple `gym_super_mario_bros` instances in parallel using `multiprocessing` for efficient data collection.
- **Monitor Wrapper**: Records `.mp4` videos of episodes that achieve new high scores, with per-worker or single-process recording options.
- **Configurable Environment**: Supports customizable frame sizes, frame stacking (default: 4 frames), frame-skipping (default: 4 frames), and action spaces (`RIGHT_ONLY`, `SIMPLE_MOVEMENT`, `COMPLEX_MOVEMENT`).
- **Robustness**: Worker processes are designed to handle failures gracefully, preventing crashes in the main process.
- **Frame Preprocessing**: Uses OpenCV for grayscale conversion and resizing, ensuring efficient input for neural networks.
- **Training and Evaluation Scripts**: Includes scripts (`train.py`, `evaluate.py`) for training RL agents and evaluating their performance.
- **Flexible Configuration**: Hyperparameters and environment settings are easily adjustable via `config.py`.

## Repository Structure

```plaintext
Mario-RL/
├── envs/
│   ├── mario_wrappers.py  # MultiMarioEnv and worker process code
│   ├── monitor.py         # Monitor wrapper for video recording
│   └── envs_test.py      # Unit tests for environment wrappers
├── agent/
│   ├── ppo.py            # PPO algorithm implementation
│   └── agent.py          # Main agent training loop
├── trained_model/        # Directory for saving trained model checkpoints
├── eval_videos/         # Directory for saving evaluation videos
├── train.py             # Main training script
├── evaluate.py          # Evaluation script for trained agents
├── config.py            # Configuration file for hyperparameters
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Requirements

### Software
- **Python**: 3.8 or 3.9 (recommended for compatibility with `gym_super_mario_bros`).
- **Operating System**: Compatible with Windows, macOS, and Linux.

### Core Dependencies
Install via `pip`:
```bash
pip install gym_super_mario_bros nes_py numpy==1.24.3 imageio opencv-python
```

### Optional Dependencies
- **PyTorch** or **TensorFlow**: For training RL models (install as needed).
- **TensorBoard**: For visualizing training progress (`pip install tensorboard`).

### Notes on Gym Compatibility
- `gym_super_mario_bros` was developed for older `gym` versions (e.g., 0.21.0) that use the legacy step API: `(obs, reward, done, info)`.
- Newer `gym` versions (>=0.26) use the updated API: `(obs, reward, terminated, truncated, info)`, which may cause compatibility issues due to passive environment checks.
- **Recommended Approach**: Use Python 3.8/3.9 with `gym==0.21.0` and `numpy==1.24.3` to avoid issues.
- **Alternative**: Use a compatibility wrapper (see [Troubleshooting](#common-problems--troubleshooting)).

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pipoiwoczz/Mario-RL.git
   cd Mario-RL
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade Packaging Tools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or install individually
   pip install gym_super_mario_bros nes_py numpy==1.24.3 imageio opencv-python
   ```

5. **Verify Installation**:
   Run the environment test script to ensure everything is set up correctly:
   ```bash
   python envs/envs_test.py
   ```

## Quick Start / Usage Examples

### Training an Agent
1. Configure hyperparameters in `config.py` (e.g., number of environments, learning rate).
2. Start training:
   ```bash
   python train.py
   ```
3. Monitor progress via logs or TensorBoard (if enabled):
   ```bash
   tensorboard --logdir=./logs
   ```

### Evaluating a Trained Model
1. Ensure a trained model is saved in `trained_model/`.
2. Run the evaluation script:
   ```bash
   python evaluate.py --model-path trained_model/ppo_model.pth
   ```
3. Check `eval_videos/` for recorded high-score videos.

### Example: Running MultiMarioEnv
```python
from envs.mario_wrappers import MultiMarioEnv

# Initialize environment with 4 workers
env = MultiMarioEnv(num_envs=4, frame_stack=4, frame_skip=4)

# Reset environment
obs = env.reset()

# Step through environment
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

# Close environment
env.close()
```

## Monitor & Recording Details

The `Monitor` wrapper records `.mp4` videos when an environment achieves a new high score. Key features:
- **Trigger**: Saves a video when `highest_reward` is exceeded for an environment.
- **Output**: Videos are saved in `eval_videos/` with filenames like `highscore_<env_id>_<score>.mp4`.
- **Customization**: Enable/disable recording per worker via `Monitor` configuration.
- **Implementation**: Uses `imageio` for efficient video encoding and OpenCV for frame processing.

To enable recording:
```python
from envs.monitor import Monitor
env = MultiMarioEnv(num_envs=4)
env = Monitor(env, save_dir="eval_videos", record=True)
```

## Training Hyperparameters

The `config.py` file allows customization of environment and training settings. Below are the key parameters:

### Environment Settings
| Parameter      | Description                                              | Default           |
|----------------|----------------------------------------------------------|-------------------|
| `NUM_ENVS`     | Number of parallel environment instances                | `4`               |
| `FRAME_STACK`  | Number of consecutive frames stacked as input state     | `4`               |
| `FRAME_SKIP`   | Number of frames skipped between actions                | `4`               |
| `FRAME_SIZE`   | Height and width of resized frames (H, W)               | `(84, 84)`        |
| `ACTION_SPACE` | Action set (`RIGHT_ONLY`, `SIMPLE_MOVEMENT`, etc.)     | `SIMPLE_MOVEMENT` |
| `SAVE_DIR`     | Directory for model checkpoints                         | `"./trained_model"` |
| `VIDEO_DIR`    | Directory for evaluation videos                         | `"./eval_videos"` |
| `LOG_PATH`     | Directory for training logs                            | `"./logs"`        |

### Training Hyperparameters
| Parameter       | Description                                                  | Default |
|-----------------|--------------------------------------------------------------|---------|
| `ROLLOUT_LEN`   | Steps per rollout before updating agent                     | `512`   |
| `MINIBATCHES`   | Number of minibatches for optimization                      | `8`     |
| `PPO_EPOCHS`    | Number of PPO update epochs per rollout                    | `7`     |
| `GAMMA`         | Discount factor for future rewards                         | `0.85`  |
| `GAE_LAMBDA`    | Lambda for Generalized Advantage Estimation                | `1.0`   |
| `CLIP_EPS`      | PPO clipping epsilon for policy update stability           | `0.3`   |
| `LR`            | Learning rate for the optimizer                            | `5e-4`  |
| `ENT_COEF`      | Coefficient for entropy bonus (exploration)                | `0.05`  |
| `VAL_COEF`      | Coefficient for value function loss                        | `0.25`  |
| `MAX_GRAD_NORM` | Maximum gradient norm for clipping                        | `0.5`   |
| `ADV_SCALE`     | Advantage scaling factor                                   | `3.0`   |

To modify these, edit `config.py` or pass arguments to `train.py`.

## Common Problems & Troubleshooting

### Gym Compatibility Issue
If using `gym>=0.26`, you may encounter errors due to the legacy step API. Use this compatibility wrapper:
```python
from gym.wrappers import EnvCompatibility
env = EnvCompatibility(MultiMarioEnv(num_envs=4))
```

### Worker Process Crashes
- **Issue**: A worker process may crash due to memory issues or emulator instability.
- **Solution**: Reduce `NUM_ENVS` or increase system resources. Ensure `numpy==1.24.3` for stability.

### Video Recording Fails
- **Issue**: Videos are not saved in `eval_videos/`.
- **Solution**: Verify `imageio` and `opencv-python` are installed. Check write permissions for `eval_videos/`.

### Slow Training
- **Issue**: Training is slow with multiple environments.
- **Solution**: Reduce `NUM_ENVS` or optimize frame preprocessing (e.g., lower `FRAME_SIZE`).

## Customization Tips

- **Action Space**: Switch between `RIGHT_ONLY`, `SIMPLE_MOVEMENT`, or `COMPLEX_MOVEMENT` in `config.py` to experiment with different control schemes.
- **Frame Processing**: Adjust `FRAME_SIZE` or `FRAME_STACK` for different input resolutions or temporal contexts.
- **RL Algorithm**: The `agent/ppo.py` implementation can be swapped for other algorithms (e.g., DQN) by modifying `agent.py`.
- **Logging**: Enable TensorBoard logging in `train.py` for detailed training metrics:
  ```python
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter(log_dir="logs")
  ```

## Acknowledgements

This project builds upon the following open-source libraries and community efforts:
- [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros) for the Mario environment.
- [nes-py](https://github.com/Kautenja/nes-py) for the NES emulator backend.
- [OpenAI Gym](https://github.com/openai/gym) for the RL environment API.
- [ImageIO](https://github.com/imageio/imageio) and [OpenCV](https://opencv.org/) for video and frame processing.
- [Super Mario Bros PPO PyTorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch) for inspiration on RL training pipelines.
