import os
import torch

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ENV = "product"

DEV_CONFIG = {
    "EPOCHS": 1,
    "K_FOLDS": 2,
    "BATCH_SIZE": 8,
    "IMAGE_WIDTH": 128,
    "IMAGE_HEIGHT": 128,
    "NUM_INFERENCE_STEPS": 20,
}

PRODUCT_CONFIG = {
    "EPOCHS": 100,
    "K_FOLDS": 5,
    "BATCH_SIZE": 48,
    "IMAGE_WIDTH": 256,
    "IMAGE_HEIGHT": 256,
    "NUM_INFERENCE_STEPS": 500,
}

CONFIG = DEV_CONFIG if ENV == "dev" else PRODUCT_CONFIG

EPOCHS = CONFIG["EPOCHS"]
K_FOLDS = CONFIG["K_FOLDS"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]
IMAGE_WIDTH = CONFIG["IMAGE_WIDTH"]
IMAGE_HEIGHT = CONFIG["IMAGE_HEIGHT"]
NUM_INFERENCE_STEPS = CONFIG["NUM_INFERENCE_STEPS"]

LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "images"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "data"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "checkpoints"))

BASE_PROMPT_PREFIX = f"Chest X-ray, radiology scan. Dx: "
BASE_PROMPT_SUFFIX = f". Clear anatomy, high-contrast grayscale."
