import os

import config
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
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device=config.DEVICE):
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

    def generate_and_save_imageV2(self, diagnose, steps=None, resolution=None):
        if steps is None:
            steps = config.NUM_INFERENCE_STEPS
        if resolution is None:
            resolution = config.IMAGE_HEIGHT

        self.pipeline.unet.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.vae.eval()

        full_prompt = f"{config.BASE_PROMPT_PREFIX}{diagnose}{config.BASE_PROMPT_SUFFIX}"

        with torch.no_grad():
            self.pipeline.to(torch.float32)
            output = self.pipeline(
                full_prompt,
                num_inference_steps=steps,
                height=resolution,
                width=resolution,
            )
        generated_image = output.images[0]
        generated_image = generated_image.convert("L")
        image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_dir = config.IMAGES_DIR
        os.makedirs(image_dir, exist_ok=True)

        file_path = os.path.join(image_dir, f"finetuned_model_generated_{image_filename}.png")
        generated_image.save(file_path)
        print(f"Diagnose: {diagnose}")
        print(f"Image saved to {file_path}")

        return file_path
    # def generate_and_save_imageV2(self, diagnose, steps=None, resolution=None):
    #     if steps is None:
    #         steps = config.NUM_INFERENCE_STEPS
    #     if resolution is None:
    #         resolution = config.IMAGE_HEIGHT
    #
    #     self.pipeline.unet.eval()
    #     self.pipeline.text_encoder.eval()
    #     self.pipeline.vae.eval()
    #
    #     full_prompt = f"{config.BASE_PROMPT_PREFIX}{diagnose}{config.BASE_PROMPT_SUFFIX}"
    #
    #     with torch.no_grad():
    #         # Set seed
    #         generator = torch.Generator(device=self.pipeline.device).manual_seed(123)
    #
    #         # Step 1: Encode prompt
    #         prompt_embeds = self.pipeline._encode_prompt(
    #             full_prompt,
    #             device=self.pipeline.device,
    #             num_images_per_prompt=1,
    #             do_classifier_free_guidance=False,
    #         )
    #
    #         # Step 2: Prepare latents (bf16 or float16)
    #         latents = self.pipeline.prepare_latents(
    #             batch_size=1,
    #             num_channels_latents=self.pipeline.unet.config.in_channels,
    #             height=resolution,
    #             width=resolution,
    #             dtype=prompt_embeds.dtype,
    #             device=self.pipeline.device,
    #             generator=generator,
    #         )
    #
    #         # Step 3: Denoising loop
    #         latents = self.pipeline.scheduler.scale_model_input(latents, 0)  # t=0 for init
    #         for t in self.pipeline.scheduler.timesteps:
    #             latent_model_input = self.pipeline.scheduler.scale_model_input(latents, t)
    #             noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
    #             latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
    #
    #         # Step 4: Convert latents to float32 before VAE decode
    #         latents = latents.to(dtype=torch.float32)
    #
    #         # Step 5: Decode
    #         image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
    #
    #         # Step 6: Post-process
    #         image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]
    #         image = image.convert("L")  # Convert to grayscale
    #
    #     # Save image
    #     image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    #     image_dir = config.IMAGES_DIR
    #     os.makedirs(image_dir, exist_ok=True)
    #     file_path = os.path.join(image_dir, f"generated_{image_filename}.png")
    #     image.save(file_path)
    #
    #     print(f"Diagnose: {diagnose}")
    #     print(f"Image saved to {file_path}")
    #
    #     return file_path

    # def generate_and_save_image(self, diagnose, steps=None, resolution=None):
    #     if steps is None:
    #         steps = config.NUM_INFERENCE_STEPS
    #     if resolution is None:
    #         resolution = config.IMAGE_HEIGHT
    #
    #     full_prompt = f"{config.BASE_PROMPT_PREFIX}{diagnose}{config.BASE_PROMPT_SUFFIX}"
    #     generated_image = self.pipeline(
    #         full_prompt,
    #         num_inference_steps=steps,
    #         height=resolution,
    #         width=resolution,
    #     ).images[0]
    #     generated_image = generated_image.convert("L")
    #     image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    #     image_dir = config.IMAGES_DIR
    #     os.makedirs(image_dir, exist_ok=True)
    #
    #     file_path = os.path.join(image_dir, f"generated_{image_filename}.png")
    #     generated_image.save(file_path)
    #     print(f"Diagnose: {diagnose}")
    #     print(f"Image saved to {file_path}")
    #
    #     return file_path

    def generate_images_Tensor(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        with torch.no_grad():
            outputs = self.pipeline(
                prompts,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                height=config.IMAGE_HEIGHT,
                width=config.IMAGE_WIDTH,
                output_type="pt",
            )
            images = outputs.images  # [batch, C, H, W]
        return images

    def load_modelV2(self, path):
        print(f"Loading model from path: {path}")
        model = XRayGenerator()
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(path, "text_encoder"),
                torch_dtype=torch.bfloat16
            ).to(self.device)

            unet = UNet2DConditionModel.from_pretrained(
                os.path.join(path, "unet"),
                torch_dtype=torch.bfloat16
            ).to(self.device)

            text_encoder.eval()
            unet.eval()

            pipeline = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                text_encoder=text_encoder,
                unet=unet,
                vae=model.vae,
                tokenizer=model.tokenizer,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            pipeline.set_progress_bar_config(disable=True)
            model.pipeline = pipeline

            print("Successfully loaded fine-tuned model components")
            return model

        except Exception as e:
            print(f"Error loading model components: {e}")
            print("Falling back to base model")
            return self

    # def load_model(self, path):
    #     text_encoder = CLIPTextModel.from_pretrained(
    #         os.path.join(path, "text_encoder"),
    #
    #     ).to(self.device)
    #     unet = UNet2DConditionModel.from_pretrained(
    #         os.path.join(path, "unet"),
    #
    #     ).to(self.device)
    #     pipeline = StableDiffusionPipeline.from_pretrained(
    #         "CompVis/stable-diffusion-v1-4",
    #     ).to(self.device)
    #     pipeline.unet = unet
    #     pipeline.text_encoder = text_encoder
    #     model = XRayGenerator()
    #     model.pipeline = pipeline
    #     return model
