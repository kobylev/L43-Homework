import torch
import os

# --- Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10

# --- Noise Configuration ---
# The noise level is a percentage of the pixel value (0.01% to 10%)
# For the math, 10% = 0.1, 0.01% = 0.0001
MIN_NOISE = 0.0001
MAX_NOISE = 0.1

# We will train on a moderate noise level, or randomly sampled noise per batch,
# let's set a default TRAINING noise level here
TRAIN_NOISE_FACTOR = 0.1

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
