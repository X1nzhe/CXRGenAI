import os
import torch

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Test config
EPOCHS = 1
K_FOLDS = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_INFERENCE_STEPS = 20

# Product config
# EPOCHS = 10
# K_FOLDS = 5
# IMAGE_WIDTH = 512
# IMAGE_HEIGHT = 512
# NUM_INFERENCE_STEPS = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "images"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "data"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "checkpoints"))

WEIGHT_DTYPE = torch.bfloat16

BASE_PROMPT_PREFIX = f"Chest X-ray, radiology scan. Dx: "
BASE_PROMPT_SUFFIX = f". Clear anatomy, high-contrast grayscale."
