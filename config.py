# Environment
WORLD = 1
STAGE = 1
ACTION_TYPE = "simple"  # "right", "simple", or "complex"

# Training
EPISODES = 20000
LR = 3e-4
GAMMA = 0.99
CLIP = 0.2
EPOCHS = 4
BATCH_SIZE = 128
MAX_STEPS = 500
SAVE_INTERVAL = 1000  # Save model every 10 episodes

# Model
INPUT_DIMS = (4, 84, 84)  # (stacked_frames, height, width)