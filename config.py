import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "images"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "data"))

CHECKPOINTS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "checkpoints"))

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

NUM_INFERENCE_STEPS = 35
