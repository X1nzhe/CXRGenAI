import os
import torch

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ENV = os.getenv("ENV", "product")

DEV_CONFIG = {
    "EPOCHS": 1,
    "K_FOLDS": 2,
    "BATCH_SIZE": 4,
    "IMAGE_WIDTH": 64,
    "IMAGE_HEIGHT": 64,
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

EPOCHS = None
K_FOLDS = None
BATCH_SIZE = None
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
NUM_INFERENCE_STEPS = None


def get_config():
    return DEV_CONFIG if ENV == "dev" else PRODUCT_CONFIG


def reload_config():
    global EPOCHS, K_FOLDS, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_INFERENCE_STEPS
    config = get_config()
    EPOCHS = config["EPOCHS"]
    K_FOLDS = config["K_FOLDS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    IMAGE_WIDTH = config["IMAGE_WIDTH"]
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"]
    NUM_INFERENCE_STEPS = config["NUM_INFERENCE_STEPS"]


reload_config()

LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "images"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "data"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "checkpoints"))

BASE_PROMPT_PREFIX = f"Chest X-ray, radiology scan. Dx: "
BASE_PROMPT_SUFFIX = f". Clear anatomy, high-contrast grayscale."
