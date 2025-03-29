import os

from peft import PeftModel

from config import IMAGES_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_INFERENCE_STEPS, WEIGHT_DTYPE, BASE_PROMPT_PREFIX, \
    BASE_PROMPT_SUFFIX
from datetime import datetime

import torch
from torch import nn
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPConfig, CLIPModel, CLIPProcessor
from transformers.utils.logging import disable_progress_bar as transformers_disable_progress_bar

from PIL import Image, ImageDraw, ImageFont
import textwrap

from models.cheXagent_evaluator import CheXagentEvaluator
from train import prepare_lora_model_for_training


def add_prompt_and_score_to_image(image, prompt, eval_score, max_chars_per_line=60):
    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    wrapped_text = textwrap.fill(prompt, max_chars_per_line)
    lines = wrapped_text.count('\n') + 1

    line_height = 20
    text_area_height = (lines + 1) * line_height + 10
    img_width, img_height = image.size
    new_img = Image.new('L', (img_width, img_height + text_area_height), color=255)
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    draw.multiline_text((10, img_height + 10), wrapped_text, font=font, fill=0)

    eval_text = f"CheXagent Eval Score: {eval_score}"
    draw.text((10, img_height + 10 + lines * line_height), eval_text, font=font, fill=0)

    return new_img


class XRayGenerator(nn.Module):
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="cuda"):
        super().__init__()
        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
        transformers_disable_progress_bar()

        self.device = device
        # self.pipeline = StableDiffusionPipeline.from_pretrained(
        #     model_name,
        # ).to(device)
        # self.pipeline.set_progress_bar_config(disable=True)
        #
        # # Default CLIP
        # self.tokenizer = self.pipeline.tokenizer
        # self.text_encoder = self.pipeline.text_encoder
        #
        # self.unet = self.pipeline.unet
        # self.vae = self.pipeline.vae

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
            width=resolution
        ).images[0]

        cheXagent = CheXagentEvaluator()
        cheXagent_score = cheXagent.evaluate_consistency(original_desc=diagnose, image=generated_image)
        img_with_text = add_prompt_and_score_to_image(generated_image, diagnose, cheXagent_score)

        image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_dir = IMAGES_DIR
        os.makedirs(image_dir, exist_ok=True)

        file_path = os.path.join(image_dir, f"generated_{image_filename}.png")
        img_with_text.save(file_path)
        print(f"Diagnose: {diagnose}")
        print(f"Image saved to {file_path}")

        return file_path, cheXagent_score

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
            outputs = self.pipeline(
                prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                output_type="pt",
            )
            images = outputs.images  # [batch, C, H, W]
        # images = (images + 1) / 2
        #
        # return images.clamp(0, 1)
        return images

    # def save_model(self, path):
    #     self.pipeline.save_pretrained(path)

    # def load_model(self, path):
    #     del self.pipeline
    #     torch.cuda.empty_cache()
    #     self.pipeline = StableDiffusionPipeline.from_pretrained(path).to(self.device)
    def load_model(self, path):
        # unet = UNet2DConditionModel.from_pretrained(path + "/unet")
        # text_encoder = CLIPTextModel.from_pretrained(path + "/text_encoder")
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(path, "text_encoder"),

        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(path, "unet"),

        ).to(self.device)
        # lora_unet = prepare_lora_model_for_training(unet)
        # lora_text_encoder = prepare_lora_model_for_training(text_encoder)
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
        ).to(self.device)
        pipeline.unet = unet
        pipeline.text_encoder = text_encoder
        model = XRayGenerator()
        model.pipeline = pipeline
        return model
    # def encode_images(self, images):
    #     with torch.no_grad():
    #         latents = self.vae.encode(images).latent_dist.sample()
    #     return latents * 0.18215
