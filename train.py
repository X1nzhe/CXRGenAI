import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import time
import numpy as np

from transformers import CLIPTextModel

from config import CHECKPOINTS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, BASE_PROMPT_PREFIX, \
    BASE_PROMPT_SUFFIX, K_FOLDS, IMAGES_DIR
from data_loader import get_dataloader


def prepare_lora_model_for_training(pipeline):
    # pipeline.unet = prepare_model_for_kbit_training(pipeline.unet)
    # pipeline.text_encoder = prepare_model_for_kbit_training(pipeline.text_encoder)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # For UNET
        modules_to_save=["conv_in"],
    )
    text_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # For Text encoder
        modules_to_save=["conv_in"],
    )
    pipeline.unet = get_peft_model(pipeline.unet, unet_lora_config)
    pipeline.text_encoder = get_peft_model(pipeline.text_encoder, text_lora_config)
    pipeline.unet.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(dtype=torch.bfloat16)
    return pipeline


class Trainer:
    def __init__(self, model, k_fold=K_FOLDS, batch_size=BATCH_SIZE, epochs=EPOCHS,
                 lr=LEARNING_RATE, checkpoint_dir=CHECKPOINTS_DIR, images_dir=IMAGES_DIR, early_stopping_patience=3):
        self.model = model
        self.model.pipeline = prepare_lora_model_for_training(model.pipeline)
        accelerator = Accelerator(mixed_precision="bf16")
        self.model.pipeline.unet, self.model.pipeline.text_encoder = accelerator.prepare(
            self.model.pipeline.unet,
            self.model.pipeline.text_encoder
        )

        self.device = model.device

        self.unet = self.model.pipeline.unet
        self.text_encoder = self.model.pipeline.text_encoder
        self.tokenizer = self.model.pipeline.tokenizer
        self.vae = self.model.pipeline.vae
        self.noise_scheduler = self.model.pipeline.scheduler

        self.unet.enable_gradient_checkpointing()
        self.text_encoder._set_gradient_checkpointing(True)
        self.unet.requires_grad_(True)
        self.text_encoder.requires_grad_(True)
        self.vae.requires_grad_(False)

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience

        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)

    def train(self):
        kfold_loaders = get_dataloader(k_folds=self.k_fold, batch_size=self.batch_size)
        ssim_metric = SSIM(data_range=2.0).to(self.device)
        psnr_metric = PSNR().to(self.device)
        train_losses, val_losses, ssim_scores, psnr_scores = [], [], [], []

        # best_val_loss = float("inf")
        best_ssim_score = float("-inf")
        best_model_info = {"fold": None, "epoch": None, "path": None}

        # Record the start time
        start_time = time.time()
        print(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} UTC")
        for fold_data in kfold_loaders:
            fold = fold_data['fold']
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']

            print(f"\nStarting training for Fold {fold}...")
            unet_lora_layers = [p for p in self.unet.parameters() if p.requires_grad]
            text_encoder_lora_layers = [p for p in self.text_encoder.parameters() if p.requires_grad]
            trainable_params = [
                {"params": unet_lora_layers, "lr": self.lr * 0.3, "weight_decay": 0.25},
                {"params": text_encoder_lora_layers, "lr": self.lr * 0.05, "weight_decay": 0.08}
            ]
            optimizer = torch.optim.AdamW(trainable_params)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=3,
                eta_min=self.lr * 0.1
            )
            torch.nn.utils.clip_grad_norm_(
                parameters=unet_lora_layers + text_encoder_lora_layers,
                max_norm=1.0
            )
            # optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)
            self.unet.train()
            self.text_encoder.train()
            early_stop_counter = 0

            for epoch in range(self.epochs):
                train_loss = self._train_epoch(train_loader, optimizer, fold, epoch)
                val_loss, ssim, psnr = self._validate_epoch(val_loader, ssim_metric, psnr_metric, fold, epoch)
                scheduler.step(train_loss)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                ssim_scores.append(ssim)
                psnr_scores.append(psnr)

                print(f"\nFold {fold} - Epoch {epoch} - Avg Train Loss: {train_loss:.4f} "
                      f"- Avg Val Loss: {val_loss:.4f} - Avg SSIM Score: {ssim:.4f} - Avg PSNR Score: {psnr:.4f}")

                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_model_info["fold"] = fold
                #     best_model_info["epoch"] = epoch
                #     best_model_info["path"] = os.path.join(
                #         self.checkpoint_dir,
                #         f"best_model_fold{fold}_epoch{epoch}"
                #     )
                #
                #     # self.model.save_model(best_model_info["path"])
                #     merged_unet = self.unet.merge_and_unload()
                #     merged_text_encoder = self.text_encoder.merge_and_unload()
                #     merged_unet.save_pretrained(os.path.join(best_model_info["path"], "unet"))
                #     merged_text_encoder.save_pretrained(os.path.join(best_model_info["path"], "text_encoder"))
                #     print(
                #         f"Best model updated: Fold {fold}, Epoch {epoch}, "
                #         f"Val Loss {val_loss:.4f}, saved to {best_model_info['path']}")
                #     early_stop_counter = 0
                if ssim > best_ssim_score:
                    best_ssim_score = ssim
                    best_model_info["fold"] = fold
                    best_model_info["epoch"] = epoch
                    best_model_info["path"] = os.path.join(
                        self.checkpoint_dir,
                        f"best_model_fold{fold}_epoch{epoch}"
                    )

                    # self.model.save_model(best_model_info["path"])
                    merged_unet = self.unet.merge_and_unload()
                    merged_text_encoder = self.text_encoder.merge_and_unload()
                    merged_unet.save_pretrained(os.path.join(best_model_info["path"], "unet"))
                    merged_text_encoder.save_pretrained(os.path.join(best_model_info["path"], "text_encoder"))
                    print(
                        f"Best model updated: Fold {fold}, Epoch {epoch}, "
                        f"Val SSIM Score {ssim:.4f}, saved to {best_model_info['path']}")
                    early_stop_counter = 0
            # checkpoint_path = os.path.join(self.checkpoint_dir, f"sd_lora_fold{fold}.pth")
            # self.model.save_model(checkpoint_path)
            # print(f"Checkpoint saved: {checkpoint_path}")
                elif early_stop_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                    break
                else:
                    early_stop_counter += 1
                    print(f"Early stopping counter: {early_stop_counter}/{self.early_stopping_patience}")
        # After training loop
        # Record the end time
        end_time = time.time()
        total_time = end_time - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} UTC")
        print(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds):.2f} seconds")
        self._plot_training_progress(train_losses, val_losses, ssim_scores, psnr_scores)

        print(f"Training complete. Running final test on the best model from {best_model_info['path']}...\n")
        finetuned_model = self.model.load_model(best_model_info["path"])
        test_loader = kfold_loaders[0]['test_loader']
        test_loss, test_ssim, test_psnr = self._test_epoch(test_loader, ssim_metric, psnr_metric)
        print(f"Final Test - Avg Test Loss: {test_loss:.4f}, Avg Test SSIM: {test_ssim:.4f}, Avg Test PSNR: {test_psnr:.4f}")
        print(f"Generating some images...")
        for batch in test_loader:
            prompts = batch['report']
            for prompt in prompts:
                finetuned_model.generate_and_save_image(prompt)
            break

    def _train_epoch(self, train_loader, optimizer, fold, epoch):
        self.unet.train()
        self.text_encoder.train()

        total_loss, num_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Training Fold {fold} - Epoch {epoch}"):
            images, texts = batch['image'], batch['report']
            images = images.to(self.device)
            loss = self._train_step(images, texts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, val_loader, ssim_metric, psnr_metric, fold, epoch):
        total_loss, total_ssim, total_psnr, num_batches = 0.0, 0.0, 0.0, 0
        self.unet.eval()
        self.text_encoder.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validating Fold {fold} - Epoch {epoch}")):
                real_images, texts = batch['image'], batch['report']

                real_images = real_images.to(self.device)
                # # debug
                # print("Real image data type:", real_images.dtype)
                # if real_images.dtype == torch.uint8:  # normalize to the range [0, 1]
                #     real_images = (real_images.float() / 255.0)
                prompts = [
                    f"{BASE_PROMPT_PREFIX}{text}{BASE_PROMPT_SUFFIX}" for text in texts
                ]
                generated_images = self.model.generate_images_Tensor(prompts)  # test new method
                # #Debug
                # print("Real image range:", real_images.min().item(), real_images.max().item())
                # print("Generated image range:", generated_images.min().item(), generated_images.max().item())

                # if generated_images.shape != real_images.shape:
                #     print(f"Shape mismatch: generated {generated_images.shape}, real {real_images.shape}")
                if epoch % 5 == 0 and batch_idx % 20 == 0:
                    self._plot_image_pair(fold, epoch, batch_idx, real_images[0:1], generated_images[0:1])

                loss = self._compute_test_loss(generated_images, real_images)

                total_loss += loss.item()
                total_ssim += ssim_metric(generated_images, real_images).item()
                total_psnr += psnr_metric(generated_images, real_images).item()
                num_batches += 1
        return total_loss / max(num_batches, 1), total_ssim / max(num_batches, 1), total_psnr / max(num_batches, 1)

    def _test_epoch(self, test_loader, ssim_metric, psnr_metric):
        total_loss, total_ssim, total_psnr, num_batches = 0.0, 0.0, 0.0, 0
        self.unet.eval()
        self.text_encoder.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing"):
                real_images, texts = batch['image'], batch['report']

                real_images = real_images.to(self.device)
                # if real_images.dtype == torch.uint8:  # normalize to the range [0, 1]
                #     real_images = (real_images.float() / 255.0)
                prompts = [
                    f"{BASE_PROMPT_PREFIX}{text}{BASE_PROMPT_SUFFIX}" for text in texts
                ]
                generated_images = self.model.generate_images_Tensor(prompts)  # test new method
                loss = self._compute_test_loss(generated_images, real_images)

                total_loss += loss.item()
                total_ssim += ssim_metric(generated_images, real_images).item()
                total_psnr += psnr_metric(generated_images, real_images).item()
                num_batches += 1
        return total_loss / max(num_batches, 1), total_ssim / max(num_batches, 1), total_psnr / max(num_batches, 1)

    def _compute_test_loss(self, generated_images, real_images):
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(generated_images, real_images)
        return loss

    def _train_step(self, images, texts):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        prompts = [
            f"{BASE_PROMPT_PREFIX}{text}{BASE_PROMPT_SUFFIX}" for text in texts
        ]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,  # 77 by default
            return_tensors="pt"
        ).to(self.device)

        text_embeddings = self.text_encoder(input_ids=text_inputs.input_ids.to(self.device),
                                            attention_mask=text_inputs.attention_mask.to(self.device)
                                            ).last_hidden_state

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device
        ).long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

    def _plot_training_progress(self, train_losses, val_losses, ssim_scores, psnr_scores):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 3, 2)
        plt.plot(ssim_scores, label="SSIM Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.legend()
        plt.title("Val SSIM Score")

        plt.subplot(1, 3, 3)
        plt.plot(psnr_scores, label="PSNR Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.legend()
        plt.title("Val PSNR Score")

        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, "training_progress.png"))
        plt.show()

    def _plot_image_pair(self, fold, epoch, batch_idx, real_image, gen_image):

        real_np = np.rot90(real_image[0, 0].cpu().numpy(), k=1)
        # real_np = np.fliplr(real_np)
        gen_np = np.rot90(gen_image[0, 0].cpu().numpy(), k=1)
        # gen_np = np.fliplr(gen_np)

        figsize = (10, 5)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(real_np, cmap='gray', origin='lower')
        axes[0].set_title(f"Fold {fold} Epoch {epoch} Batch {batch_idx} Sample 0 Real Image")
        axes[0].axis('off')

        axes[1].imshow(gen_np, cmap='gray', origin='lower')
        axes[1].set_title(f"Fold {fold} Epoch {epoch} Batch {batch_idx} Sample 0 Generated Image")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, f"fold{fold}_epoch{epoch}_batch{batch_idx}_comparison.png"))
        plt.show()


    # def _save_model(self, path):
    #     model = self.model.merge_and_unload()
    #     model.save_pretrained(path)


