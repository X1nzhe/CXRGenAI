# Final Project Assignment – Applied Deep Learning and Generative AI in Healthcare
## Track 2: Text to Medical Image Synthesis (Image generation from Descriptions)
### Introduction: 
This project aims to develop a diffusion-based deep learning model 
for generating synthetic X-ray images from textual descriptions. 
By training the model, it enables the automatic generation of medical images from diagnostic inputs.
### Features
#### Text-to-X-ray with Optimized Stable Diffusion
Generates synthetic X-ray images from textual descriptions by adapting Stable Diffusion with latent space optimization, improving image realism and medical applicability.
#### Transfer Learning
Leverages pre-trained models and fine-tuning techniques to improve performance on limited medical datasets.
#### LoRA Fine-Tuning
Efficiently fine-tunes both the text encoder and UNet using Low-Rank Adaptation (LoRA) for domain-specific medical imaging tasks.
#### Optimization Strategy
- Uses AdamW optimizer with weight decay for stable convergence.
- Implements a Cosine Annealing Learning Rate Scheduler for smooth decay.
- Applies gradient norm clipping to prevent gradient explosion and stabilize training.
#### Robust Model Evaluation
Tracks performance with MSE (Mean Squared Error) loss, SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio), integrating an early stopping mechanism based on SSIM.
#### Cross-Validation
Uses an 80/20 train-test split with 5-fold cross-validation for comprehensive model evaluation.
#### Automated Dataset Handling
Implements multi-threaded dataset downloading and preprocessing for efficient training.
#### Optimized for Google Colab
Fully tested with A100/L4 GPU acceleration, unleashing performance of the hardware, ensuring smooth training and inference.
### Instructions to run the code

### Compatibility Notice
The project is independently runnable; 
however, it has been tested and trained on Google Colab, 
relying on Colab’s specific environment and hardware accelerations (e.g., GPU/TPU). 
Running it on other platforms may require additional setup, 
and proper functionality cannot be guaranteed.
