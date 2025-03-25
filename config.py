import os
import torch

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "images"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "data"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "checkpoints"))

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

NUM_INFERENCE_STEPS = 20

BASE_PROMPT_PREFIX = f"Chest X-ray, radiology scan. Dx: "
BASE_PROMPT_SUFFIX = f". Clear anatomy, high-contrast grayscale."
