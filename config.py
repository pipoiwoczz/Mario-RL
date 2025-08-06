# %%writefile config.py # for colab only

# Environment
WORLD = 1
STAGE = 1
ACTION_TYPE = "simple"  # "right", "simple", or "complex"

# Training
EPISODES = 10000
MAX_STEPS = 200
BATCH_SIZE = 32  # Larger batches for stability
SAVE_INTERVAL = 100  # Save every 100 episodes
LOAD_EPISODE = 0  # Episode to load (0 for new training)
TOTAL_MAX_STEPS = 1e6

# Model
INPUT_DIMS = (4, 84, 84)  # Stacked frames
LR = 3e-4
GAMMA = 0.99
CLIP = 0.2
ENTROPY_COEF = 0.01

# Paths
# DRIVE_DIR = "/content/drive/MyDrive/mario_checkpoints_4.0"
DRIVE_DIR = '/checkpoints/'
