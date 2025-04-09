# Final Project Assignment – Applied Deep Learning and Generative AI in Healthcare
### Track 2: Text to Medical Image Synthesis (Image generation from Descriptions)
## Introduction 
This project aims to develop a diffusion-based deep learning model 
for generating synthetic X-ray images from textual descriptions. 
By training the model, it enables the automatic generation of medical images from diagnostic inputs.
## Compatibility Notice
>The project is independently runnable; 
>however, it has been tested on Google Colab, 
>relying on Colab’s specific environment and hardware accelerations (e.g., A100 GPU or L4 GPU). 
>Running it on other platforms may require additional setup.
## Instructions to run the code (in Google Colab)
>❗️❗️ Two ready-to-use Jupyter notebooks (*_Example_Usage.ipynb) are provided for your convenience. 

>❗️❗️ In Google Colab, Set the runtime type to L4 GPU or A100 GPU. Other hardware accelerations may not be supported.
### Usage 1: Model Training
#### 0. Download code from Github
```
!git clone https://github.com/X1nzhe/CXRGenAI.git
```
#### 1. Install Required Dependencies
```
!pip install -q -r /content/CXRGenAI/requirements.txt
```

#### 2. Quickly Run in Development or Debug Configuration
```
# In Development/Debug configuration, hyperparameters are set to minimal values to allow quick execution.

# DEV_CONFIG = {
    "EPOCHS": 1,
    "K_FOLDS": 2,
    "BATCH_SIZE": 8,
    "IMAGE_WIDTH": 128,
    "IMAGE_HEIGHT": 128,
    "NUM_INFERENCE_STEPS": 20,
}
```

```
# Command-line Arguments 
# --env: Runtime environment. By default, it set to 'product'. Another option is 'dev'
# --mode: The mode of how would you like to use the software, chosing 'train' or 'generate'
# --model_path: Path to pre-trained model（'generate' mode only)
# --description: Text description to generate X-Ray image（'generate' mode only)
```
```
# Set '--env' to dev for the using of debug config
!python /content/CXRGenAI/main.py --env dev --mode train
```

```
import os
import glob
from IPython.display import display
from PIL import Image

def show_images_from(image_dir):
    png_files = glob.glob(os.path.join(image_dir, "*.png"))

    if not png_files:
        return

    for img_path in sorted(png_files):
        img = Image.open(img_path)
        display(img)

show_images_from("/content/CXRGenAI/images")
```
#### 3. Fully Run for Training the model
```
# (Optinal) Remove images directory if you were run the code in Debug mode
!rm -rf /content/CXRGenAI/images
```
```
# Hyperparameters are set to normal values to allow fully training in Product configuration. 
# However, it takes 10+ hours for fully training.

# PRODUCT_CONFIG = {
    "EPOCHS": 100,
    "K_FOLDS": 5,
    "BATCH_SIZE": 48,
    "IMAGE_WIDTH": 256,
    "IMAGE_HEIGHT": 256,
    "NUM_INFERENCE_STEPS": 500,
}
```
```
# Command-line Arguments 
# --env: Runtime environment. By default, it set to 'product'. Another option is 'dev'
# --mode: The mode of how would you like to use the software, chosing 'train' or 'generate'
# --model_path: Path to pre-trained model（'generate' mode only)
# --description: Text description to generate X-Ray image（'generate' mode only)
```
```
!python /content/CXRGenAI/main.py --mode train
```
#### 4. Review results

```
import os
import glob
from IPython.display import display
from PIL import Image

def show_images_from(image_dir):
    png_files = glob.glob(os.path.join(image_dir, "*.png"))

    if not png_files:
        return

    for img_path in sorted(png_files):
        img = Image.open(img_path)
        display(img)

show_images_from("/content/CXRGenAI/images")
```
#### 5. Save the fine-tuned model and training results
```
import shutil

# The best model will be printed out after training completed.
# e.g. Training complete. Running final test on the best model from /content/CXRGenAI/checkpoints/best_model_fold3_epoch2...

# The pretrained model was saved as best_model_foldX_epochY.zip (where X and Y represent digits) 
best_model = "best_model_foldX_epochY"
shutil.make_archive(f"/content/{best_model}", 'zip', f"/content/CXRGenAI/checkpoints/{best_model}")
```
```
from google.colab import files

files.download(f"/content/{best_model}.zip")
```
```
shutil.make_archive("/content/images", 'zip', "/content/CXRGenAI/images/")
files.download("/content/images.zip")
```
```
from google.colab import drive
drive.mount('/content/drive')

import shutil
# The pretrained model was saved as best_model_foldX_epochY.zip (where X and Y represent digits) 
best = "/content/best_model_foldX_epochY.zip"
shutil.copy(best, "/content/drive/PATH_TO_DIR")
```
### Usage 2: Xray Image Generation with Pretrained Model
#### 0. Upload the pretrained model into Google Drive
The pretrained model was saved as **best_model_foldX_epochY.zip** (where X and Y represent digits), e.g. best_model_fold3_epoch2.zip 
#### 1. Download code from Github
```
!git clone https://github.com/X1nzhe/CXRGenAI.git
```
#### 2. Install Required Dependencies
```
!pip install -q -r /content/CXRGenAI/requirements.txt
```
#### 3. Load pre-trained model
```
from google.colab import drive
drive.mount('/content/drive')

zip_path = "/content/drive/PATH_TO_ZIP_FILE_OF_MODEL"
extract_folder = "/content/CXRGenAI/checkpoints/best_model"

import shutil
shutil.unpack_archive(zip_path, extract_folder)
```
#### 4. Generate Xray images
```
model_path = "/content/CXRGenAI/checkpoints/best_model"

description = "The cardiomediastinal silhouette is within normal limits, with no evidence of enlargement or mass. The lungs show no signs of pneumothorax or significant opacities. Mild interstitial markings are noted. There are no large pleural effusions. No acute findings."

!python /content/CXRGenAI/main.py --env product --mode generate --model_path {model_path} --description "{description}"
```
#### 5. Dispaly Result
```
import os
import glob
from IPython.display import display
from PIL import Image

generated_img_path = "/content/CXRGenAI/images/PATH_TO_GENERATED_XRAY_IMAGE"

display(Image.open(generated_img_path))
```


## Features
- #### Text-to-X-ray generation
    Generates synthetic X-ray images from textual descriptions by adapting Stable Diffusion with latent space optimization, improving image realism and medical applicability.
- #### Automated Dataset Handling
    Automated downloading and preprocessing dataset with multi-threaded acceleration for efficient training.
- #### Optimized for Google Colab
    Fully tested with A100/L4 GPU acceleration, unleashing performance of the hardware, ensuring smooth training and inference.
## Tech
- _**Transfer Learning:**_ Leverages pre-trained models and fine-tuning techniques to improve performance on limited medical datasets.
- _**LoRA Fine-Tuning:**_ Efficiently fine-tunes both the text encoder and UNet using Low-Rank Adaptation (LoRA) for domain-specific medical imaging tasks.
- _**AdamW Optimizer:**_ Uses AdamW optimizer with weight decay for stable convergence. 
- _**Cosine Annealing LR Scheduler:**_  Deploys a Cosine Annealing Learning Rate Scheduler for smooth decay. 
- _**Gradient Norm Clipping:**_ Applies gradient norm clipping to prevent gradient explosion and stabilize training.
- _**Evaluation Strategy:**_ Tracks model performance with MSE (Mean Squared Error) loss, SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio), integrating an early stopping mechanism based on SSIM.
- _**Early Stopping:**_ Stops training early if SSIM plateaus to prevent overfitting and save computation resources.
- _**Cross-Validation:**_ Uses an 80/20 train-test split with 5-fold cross-validation for comprehensive model evaluation.

