import torch
from diffusers import StableDiffusionPipeline
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR
from tqdm import tqdm
import config


def load_baseline_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


class BaselineEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ssim_metric = SSIM(data_range=2.0).to(device)
        self.psnr_metric = PSNR().to(device)

    def evaluate(self, dataloader):
        total_loss, total_ssim, total_psnr, num_batches = 0.0, 0.0, 0.0, 0
        self.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Baseline"):
                real_images, texts = batch['image'], batch['report']
                real_images = real_images.to(self.device)

                prompts = [
                    f"{config.BASE_PROMPT_PREFIX}{text}{config.BASE_PROMPT_SUFFIX}" for text in texts
                ]

                outputs = self.model(
                    prompts,
                    num_inference_steps=config.NUM_INFERENCE_STEPS,
                    height=config.IMAGE_HEIGHT,
                    width=config.IMAGE_WIDTH,
                    output_type="pt",
                )
                generated_images = outputs.images

                loss = torch.nn.functional.mse_loss(generated_images, real_images)
                total_loss += loss.item()
                total_ssim += self.ssim_metric(generated_images, real_images).item()
                total_psnr += self.psnr_metric(generated_images, real_images).item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_ssim = total_ssim / num_batches
        avg_psnr = total_psnr / num_batches

        print(f"\n[Baseline Evaluation] Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")
        return avg_loss, avg_ssim, avg_psnr