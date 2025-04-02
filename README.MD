# Final Project Assignment – Applied Deep Learning and Generative AI in Healthcare
## Track 2: Text to Medical Image Synthesis (Image generation from Descriptions)
### Introduction: 
This project aims to develop a diffusion-based deep learning model 
for generating synthetic X-ray images from textual descriptions. 
By training the model, it enables the automatic generation of medical images from diagnostic inputs.
### Features
- #### Text-to-X-ray generation with Optimized Stable Diffusion
    Generates synthetic X-ray images from textual descriptions by adapting Stable Diffusion with latent space optimization, improving image realism and medical applicability.
- #### Automated Dataset Handling
    Automated downloading and preprocessing dataset with multi-threaded acceleration for efficient training.
- #### Optimized for Google Colab
    Fully tested with A100/L4 GPU acceleration, unleashing performance of the hardware, ensuring smooth training and inference.
### Tech
- _**Transfer Learning:**_ Leverages pre-trained models and fine-tuning techniques to improve performance on limited medical datasets.
- _**LoRA Fine-Tuning:**_ Efficiently fine-tunes both the text encoder and UNet using Low-Rank Adaptation (LoRA) for domain-specific medical imaging tasks.
- _**AdamW Optimizer:**_ Uses AdamW optimizer with weight decay for stable convergence. 
- _**Cosine Annealing LR Scheduler:**_  Deploys a Cosine Annealing Learning Rate Scheduler for smooth decay. 
- _**Gradient Norm Clipping:**_ Applies gradient norm clipping to prevent gradient explosion and stabilize training.
- _**Evaluation Strategy:**_ Tracks model performance with MSE (Mean Squared Error) loss, SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio), integrating an early stopping mechanism based on SSIM.
- _**Cross-Validation:**_ Uses an 80/20 train-test split with 5-fold cross-validation for comprehensive model evaluation.

### Instructions to run the code

### Compatibility Notice
>The project is independently runnable; 
>however, it has been tested and trained on Google Colab, 
>relying on Colab’s specific environment and hardware accelerations (e.g., A100 GPU or L4 GPU). 
>Running it on other platforms may require additional setup, 
>and proper functionality cannot be guaranteed.
