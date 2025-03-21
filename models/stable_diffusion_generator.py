import os
from config import IMAGES_DIR
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from peft import get_peft_model, LoraConfig, TaskType
import tqdm

from PIL import Image, ImageDraw, ImageFont
import textwrap

from utils.data_loader import get_dataloader


def add_prompt_to_image(image, prompt, max_chars_per_line=60):
    wrapped_text = textwrap.fill(prompt, max_chars_per_line)
    lines = wrapped_text.count('\n') + 1

    line_height = 20
    text_area_height = lines * line_height + 20
    img_width, img_height = image.size
    new_img = Image.new('L', (img_width, img_height + text_area_height), color=(255, 255, 255))
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    draw.multiline_text((10, img_height + 10), wrapped_text, font=font, fill=(0, 0, 0))

    return new_img


class XRayGenerator:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="cuda"):
        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
        self.pipeline.set_progress_bar_config(disable=True)

        # LoRA
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        self.pipeline.unet = get_peft_model(self.pipeline.unet, self.lora_config)
    #
    # def train(self, k_fold=5, batch_size=8, epochs=5, lr=1e-4, checkpoint_dir="../checkpoints"):
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     kfold_loaders = get_dataloader(k_folds=k_fold, batch_size=batch_size)
    #     ssim_metric = SSIM(data_range=1.0).to(self.device)
    #
    #     for fold_data in kfold_loaders:
    #         fold = fold_data['fold']
    #         train_loader = fold_data['train_loader']
    #         val_loader = fold_data['val_loader']
    #
    #         print(f"\nStarting training for Fold {fold}...")
    #         optimizer = torch.optim.AdamW(self.pipeline.unet.parameters(), lr=lr)
    #         self.pipeline.unet.train()
    #
    #         for epoch in range(epochs):
    #             total_train_loss = 0.0
    #             num_train_batches = 0
    #
    #             # Training step
    #             for images, texts in tqdm(train_loader, desc=f"Training Fold {fold} - Epoch {epoch + 1}"):
    #                 images = images.to(self.device)
    #                 loss = self._train_step(images, texts)
    #
    #                 loss.backward()
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #
    #                 total_train_loss += loss.item()
    #                 num_train_batches += 1
    #
    #             avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
    #
    #             # Validation step
    #             total_val_loss = 0.0
    #             total_ssim = 0.0
    #             num_val_batches = 0
    #             self.pipeline.unet.eval()
    #
    #             with torch.no_grad():
    #                 for real_images, texts in tqdm(val_loader, desc=f"Validating Fold {fold} - Epoch {epoch + 1}"):
    #                     real_images = real_images.to(self.device)
    #                     generated_images = self.generate_images_for_ssim(texts)
    #
    #                     # Calculate Loss
    #                     loss = self._train_step(real_images, texts)
    #                     total_val_loss += loss.item()
    #
    #                     # Calculate SSIM
    #                     ssim_score = ssim_metric(generated_images, real_images)
    #                     total_ssim += ssim_score.item()
    #
    #                     num_val_batches += 1
    #
    #             avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    #             avg_ssim = total_ssim / num_val_batches if num_val_batches > 0 else 0.0
    #
    #             print(f"Fold {fold} - Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} "
    #                   f"- Val Loss: {avg_val_loss:.4f} - SSIM: {avg_ssim:.4f}")
    #
    #
    #     checkpoint_path = os.path.join(checkpoint_dir, f"sd_lora_fold{fold}.pth")
    #     self.save_model(checkpoint_path)
    #     print(f"Checkpoint saved: {checkpoint_path}")
    #
    # def _train_step(self, images, texts):
    #     return self.pipeline.unet(images, texts.input_ids).loss

    def generate_and_save_image(self, prompt):
        generated_image = self.pipeline(prompt).images[0]
        img_with_text = add_prompt_to_image(generated_image, prompt)
        # TODO: ADD cheXagent score to image

        image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_dir = IMAGES_DIR
        os.makedirs(image_dir, exist_ok=True)

        file_path = os.path.join(image_dir, f"generated_{image_filename}.png")

        counter = 1
        original_file_path = file_path

        while os.path.exists(file_path):
            file_path = f"{original_file_path.stem}_{counter}.png"
            counter += 1

        img_with_text.save(file_path)
        print(f"Image saved to {file_path}")

        return file_path

    def generate_images_for_ssim(self, prompts):
        with torch.no_grad():
            images = self.pipeline(prompts).images

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        processed_images = []
        for img in images:
            processed_image = transform(img).unsqueeze(0).to(self.device)
            processed_images.append(processed_image)

        # Stack all images into a single tensor with shape (batch_size, channels, height, width)
        return torch.cat(processed_images, dim=0)

    def save_model(self, path):
        self.pipeline.unet.save_pretrained(path)

    def load_model(self, path):
        self.pipeline.unet.load_pretrained(path)
