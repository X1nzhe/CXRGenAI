import os
from config import IMAGES_DIR
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging

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
    new_img = Image.new('L', (img_width, img_height + text_area_height), color=255)
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    draw.multiline_text((10, img_height + 10), wrapped_text, font=font, fill=0)

    return new_img


class XRayGenerator:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="cuda"):
        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()

        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        self.pipeline.set_progress_bar_config(disable=True)

    def generate_and_save_image(self, prompt):
        generated_image = self.pipeline(
                prompt,
                num_inference_steps=50,
                height=256,
                width=256
            ).images[0]
        img_with_text = add_prompt_to_image(generated_image, prompt)
        # TODO: ADD cheXagent score to image

        image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_dir = IMAGES_DIR
        os.makedirs(image_dir, exist_ok=True)

        file_path = os.path.join(image_dir, f"generated_{image_filename}.png")
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

    def generate_images_for_ssimV2(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.pipeline(
                    prompts,
                    num_inference_steps=50,
                    height=256,
                    width=256,
                    output_type="pt",
                )
                images = outputs.images  # [batch, 1, 256, 256]
        return images.clamp(0, 1)

    def save_model(self, path):
        self.pipeline.unet.save_pretrained(path)

    def load_model(self, path):
        self.pipeline.unet.from_pretrained(path)

    # def encode_images(self, images):
    #     with torch.no_grad():
    #         latents = self.vae.encode(images).latent_dist.sample()
    #     return latents * 0.18215
