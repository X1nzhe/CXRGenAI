import os

from config import IMAGES_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_INFERENCE_STEPS, BASE_PROMPT_PREFIX, \
    BASE_PROMPT_SUFFIX, DEVICE
from datetime import datetime

import torch
from torch import nn

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.utils.logging import disable_progress_bar as transformers_disable_progress_bar

from PIL import Image, ImageDraw, ImageFont
import textwrap


def add_prompt_to_image(image, prompt, max_chars_per_line=60):
    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    wrapped_text = textwrap.fill(prompt, max_chars_per_line)
    lines = wrapped_text.count('\n') + 1

    line_height = 20
    text_area_height = lines * line_height + 10
    img_width, img_height = image.size
    new_img = Image.new('L', (img_width, img_height + text_area_height), color=255)
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    draw.multiline_text((10, img_height + 10), wrapped_text, font=font, fill=0)

    return new_img


class XRayGenerator(nn.Module):
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device=DEVICE):
        super().__init__()
        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
        transformers_disable_progress_bar()

        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        ).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16
        ).to(device)

        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(device)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            unet=self.unet,
            text_encoder=self.text_encoder,
            vae=self.vae,
            tokenizer=self.tokenizer
        ).to(device)

        self.pipeline.set_progress_bar_config(disable=True)

    def generate_and_save_image(self, diagnose, steps=NUM_INFERENCE_STEPS, resolution=IMAGE_HEIGHT):
        full_prompt = f"{BASE_PROMPT_PREFIX}{diagnose}{BASE_PROMPT_SUFFIX}"
        generated_image = self.pipeline(
            full_prompt,
            num_inference_steps=steps,
            height=resolution,
            width=resolution,
        ).images[0]
        generated_image = generated_image.convert("L")
        image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_dir = IMAGES_DIR
        os.makedirs(image_dir, exist_ok=True)

        file_path = os.path.join(image_dir, f"generated_{image_filename}.png")
        generated_image.save(file_path)
        print(f"Diagnose: {diagnose}")
        print(f"Image saved to {file_path}")

        return file_path

    def generate_images_Tensor(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        with torch.no_grad():
            outputs = self.pipeline(
                prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                output_type="pt",
            )
            images = outputs.images  # [batch, C, H, W]
        return images

    def load_model(self, path):
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(path, "text_encoder"),

        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(path, "unet"),

        ).to(self.device)
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
        ).to(self.device)
        pipeline.unet = unet
        pipeline.text_encoder = text_encoder
        model = XRayGenerator()
        model.pipeline = pipeline
        return model
