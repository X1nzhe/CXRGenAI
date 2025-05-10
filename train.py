import copy
import os

import optuna
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig
import time
import numpy as np
import config
from data_loader import get_dataloader
from stable_diffusion_baseline import BaselineEvaluator, load_baseline_pipeline
from datetime import datetime
import textwrap

from stable_diffusion_generator import XRayGenerator


# def prepare_lora_model_for_training(pipeline):
#     unet_lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.1,
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # For UNET
#         modules_to_save=["conv_in"],
#     )
#     text_lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # For Text encoder
#         modules_to_save=["conv_in"],
#     )
#     pipeline.unet = get_peft_model(pipeline.unet, unet_lora_config)
#     pipeline.text_encoder = get_peft_model(pipeline.text_encoder, text_lora_config)
#     pipeline.unet.to(dtype=torch.bfloat16)
#     pipeline.text_encoder.to(dtype=torch.bfloat16)
#     return pipeline

def prepare_lora_model_for_trainingV2(pipeline, unet_config, text_config):
    unet_lora_config = LoraConfig(
        r=unet_config["r"],
        lora_alpha=unet_config["alpha"],
        lora_dropout=unet_config["dropout"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # For UNET
    )
    text_lora_config = LoraConfig(
        r=text_config["r"],
        lora_alpha=text_config["alpha"],
        lora_dropout=text_config["dropout"],
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # For Text encoder
    )
    pipeline.unet = get_peft_model(pipeline.unet, unet_lora_config)
    pipeline.text_encoder = get_peft_model(pipeline.text_encoder, text_lora_config)
    pipeline.unet.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(dtype=torch.bfloat16)
    return pipeline


def concat_images_with_prompt(finetuned_image_path, baseline_image_path, prompt):
    image_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    file_path = os.path.join(config.IMAGES_DIR, f"comparison_{image_filename}.png")

    img1 = Image.open(finetuned_image_path).convert("L")
    img2 = Image.open(baseline_image_path).convert("L")

    if img1.height != img2.height:
        new_height = max(img1.height, img2.height)
        img1 = img1.resize((img1.width, new_height))
        img2 = img2.resize((img2.width, new_height))

    img1_np = np.array(img1)
    img2_np = np.array(img2)
    wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=100))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1_np, cmap="gray")
    axes[0].set_title("Fine-tuned Model", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img2_np, cmap="gray")
    axes[1].set_title("Baseline Model", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(f"Diagnose: {wrapped_prompt}", fontsize=12, y=1.05, ha='left', x=0.1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image to {file_path}")


class Trainer:
    def __init__(self, model, k_fold=None, batch_size=None, epochs=None, unet_lora_config=None, text_lora_config=None,
                 scheduler_config=None, lr_unet=None, lr_text=None, wd_unet=None, wd_text=None, checkpoint_dir=None,
                 images_dir=None, early_stopping_patience=5, for_hpo=False, max_trial_time=300, trial=None):

        self.for_hpo = for_hpo
        self.trial = trial
        self.model = model
        self.unet_lora_config = unet_lora_config if unet_lora_config is not None else {"r": 8, "alpha": 16,
                                                                                       "dropout": 0.1}
        self.text_lora_config = text_lora_config if text_lora_config is not None else {"r": 8, "alpha": 12,
                                                                                       "dropout": 0.05}
        self.scheduler_config = scheduler_config if scheduler_config is not None else {"T_max": 5, "eta_min": config.LEARNING_RATE*0.001}

        # self.model.pipeline = prepare_lora_model_for_trainingV2(model.pipeline, self.unet_lora_config, self.text_lora_config)
        # accelerator = Accelerator(mixed_precision="bf16")
        # self.model.pipeline.unet, self.model.pipeline.text_encoder = accelerator.prepare(
        #     self.model.pipeline.unet,
        #     self.model.pipeline.text_encoder
        # )
        self._prepare_model_for_training()

        self.device = model.device

        # self.unet = self.model.pipeline.unet
        # self.text_encoder = self.model.pipeline.text_encoder
        # self.tokenizer = self.model.pipeline.tokenizer
        # self.vae = self.model.pipeline.vae
        # self.noise_scheduler = self.model.pipeline.scheduler
        #
        # self.unet.enable_gradient_checkpointing()
        # if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
        #     self.text_encoder.gradient_checkpointing_enable()
        # self.unet.requires_grad_(True)
        # self.text_encoder.requires_grad_(True)
        # self.vae.requires_grad_(False)

        self.k_fold = k_fold if k_fold is not None else config.K_FOLDS
        self.batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
        self.epochs = epochs if epochs is not None else config.EPOCHS
        self.lr_unet = lr_unet if lr_unet is not None else config.LEARNING_RATE * 0.1
        self.lr_text = lr_text if lr_text is not None else config.LEARNING_RATE * 0.02
        self.wd_unet = wd_unet if wd_unet is not None else 0.1
        self.wd_text = wd_text if wd_text is not None else 0.05

        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else config.CHECKPOINTS_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.images_dir = images_dir if images_dir is not None else config.IMAGES_DIR
        os.makedirs(self.images_dir, exist_ok=True)

        self.early_stopping_patience = early_stopping_patience
        self.max_trial_time = max_trial_time

        # self.unet_lora_layers = [p for p in self.unet.parameters() if p.requires_grad]
        # self.text_encoder_lora_layers = [p for p in self.text_encoder.parameters() if p.requires_grad]
        self._set_trainable_layers()

    def _prepare_model_for_training(self):
        self.model.pipeline = prepare_lora_model_for_trainingV2(self.model.pipeline, self.unet_lora_config,
                                                                self.text_lora_config)
        accelerator = Accelerator(mixed_precision="bf16")
        self.model.pipeline.unet, self.model.pipeline.text_encoder = accelerator.prepare(
            self.model.pipeline.unet,
            self.model.pipeline.text_encoder
        )

        self.unet = self.model.pipeline.unet
        self.text_encoder = self.model.pipeline.text_encoder
        self.tokenizer = self.model.pipeline.tokenizer
        self.vae = self.model.pipeline.vae
        self.noise_scheduler = self.model.pipeline.scheduler

        self.unet.enable_gradient_checkpointing()
        if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
            self.text_encoder.gradient_checkpointing_enable()

        self.unet.requires_grad_(True)
        self.text_encoder.requires_grad_(True)
        self.vae.requires_grad_(False)

    def _set_trainable_layers(self):
        self.unet_lora_layers = [p for p in self.unet.parameters() if p.requires_grad]
        self.text_encoder_lora_layers = [p for p in self.text_encoder.parameters() if p.requires_grad]

    def reset_model(self):
        print("\nResetting model to initial pre-trained state...")

        del self.model.pipeline
        del self.unet
        del self.text_encoder
        del self.tokenizer
        del self.vae
        del self.noise_scheduler
        torch.cuda.empty_cache()

        self.model = XRayGenerator()

        self._prepare_model_for_training()

        self._set_trainable_layers()

        print("Model reset complete.")

    def train(self):
        kfold_loaders = get_dataloader(k_folds=self.k_fold, batch_size=self.batch_size)
        ssim_metric = SSIM(data_range=1.0).to(self.device)
        psnr_metric = PSNR(data_range=1.0).to(self.device)
        train_losses, val_losses, ssim_scores, psnr_scores = [], [], [], []

        best_ssim_score = float("-inf")
        best_model_info = {"fold": None, "epoch": None, "path": None}

        # Record the start time
        start_time = time.time()
        print(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} UTC")
        fold_range = kfold_loaders[:1] if self.for_hpo else kfold_loaders

        for fold_idx, fold_data in enumerate(fold_range):

            if fold_idx > 0:
                self.reset_model()

            fold = fold_data['fold']
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']

            if self.k_fold > 1:
                print(f"\nStarting training for Fold {fold}...")
            else:
                print(f"\nStarting training...")
            trainable_params = [
                {"params": self.unet_lora_layers, "lr": self.lr_unet, "weight_decay": self.wd_unet},
                {"params":self.text_encoder_lora_layers, "lr": self.lr_text, "weight_decay": self.wd_text}
            ]
            optimizer = torch.optim.AdamW(trainable_params)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config["T_max"],
                eta_min=self.scheduler_config["eta_min"]
            )

            self.unet.train()
            self.text_encoder.train()
            early_stop_counter = 0

            for epoch in range(self.epochs):
                train_loss = self._train_epoch(train_loader, optimizer, fold, epoch)
                scheduler.step()
                train_losses.append(train_loss)
                val_loss, ssim, psnr = self._validate_epoch(val_loader, ssim_metric, psnr_metric, fold, epoch)
                val_losses.append(val_loss)
                ssim_scores.append(ssim)
                psnr_scores.append(psnr)
                if self.k_fold > 1:
                    print(f"\nFold {fold} - Epoch {epoch} - Avg Train Loss: {train_loss:.4f} "
                          f"- Avg Val Loss: {val_loss:.4f} - Avg SSIM Score: {ssim:.4f} - Avg PSNR Score: {psnr:.4f}")
                else:
                    print(f"\nEpoch {epoch} - Avg Train Loss: {train_loss:.4f} "
                          f"- Avg Val Loss: {val_loss:.4f} - Avg SSIM Score: {ssim:.4f} - Avg PSNR Score: {psnr:.4f}")

                if ssim > best_ssim_score:
                    best_ssim_score = ssim
                    best_model_info["fold"] = fold
                    best_model_info["epoch"] = epoch
                    best_model_info["path"] = os.path.join(
                        self.checkpoint_dir,
                        f"best_model_fold{fold}_epoch{epoch}"
                    )
                    # merged_unet = self.unet.merge_and_unload()
                    # merged_text_encoder = self.text_encoder.merge_and_unload()
                    # merged_unet.save_pretrained(os.path.join(best_model_info["path"], "unet"))
                    # merged_text_encoder.save_pretrained(os.path.join(best_model_info["path"], "text_encoder"))
                    self.save_best_model(best_model_info["path"])
                    if self.k_fold > 1:
                        print(f"Best model updated: Fold {fold}, Epoch {epoch}, "
                            f"Val SSIM Score {ssim:.4f}, saved to {best_model_info['path']}")
                    else:
                        print(f"Best model updated: Epoch {epoch}, "
                              f"Val SSIM Score {ssim:.4f}, saved to {best_model_info['path']}")
                    early_stop_counter = 0

                elif early_stop_counter >= self.early_stopping_patience:
                    print(
                        f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                    break
                else:
                    early_stop_counter += 1
                    print(f"Early stopping counter: {early_stop_counter}/{self.early_stopping_patience}")

                if self.trial:
                    self.trial.report(train_loss, step=epoch)
                    if self.trial.should_prune():
                        print(f"Trial pruned at epoch {epoch} with Val loss {train_loss:.4f}")
                        raise optuna.exceptions.TrialPruned()
            if self.for_hpo:
                return best_ssim_score

        # After training loop
        # merged_unet = self.unet.merge_and_unload()
        # merged_text_encoder = self.text_encoder.merge_and_unload()
        # merged_unet.save_pretrained(os.path.join(best_model_info["path"], "unet"))
        # merged_text_encoder.save_pretrained(os.path.join(best_model_info["path"], "text_encoder"))
        # Record the end time
        end_time = time.time()
        total_time = end_time - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} UTC")
        print(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds):.2f} seconds")
        self._plot_training_progressV2(train_losses, val_losses, ssim_scores, psnr_scores)

        print(f"Training complete. Running final test on the best model from {best_model_info['path']}...\n")
        finetuned_model = self.model.load_modelV2(best_model_info["path"])
        test_loader = kfold_loaders[0]['test_loader']
        test_loss, test_ssim, test_psnr = self._test_epoch(test_loader, ssim_metric, psnr_metric)

        baseline_model_pipe = load_baseline_pipeline().to(self.device)
        baseline_model = BaselineEvaluator(baseline_model_pipe, self.device)
        baseline_loss, baseline_ssim, baseline_psnr = baseline_model.evaluate(test_loader)

        print(
            f"Final Test - Fine-tuned Model- Avg Test Loss: {test_loss:.4f}, Avg Test SSIM: {test_ssim:.4f}, Avg Test PSNR: {test_psnr:.4f}")
        print(
            f"Final Test - Baseline Model - Avg Test Loss: {baseline_loss:.4f}, Avg Test SSIM: {baseline_ssim:.4f}, Avg Test PSNR: {baseline_psnr:.4f}")

        print(f"Generating some images...")
        for batch in test_loader:
            prompts = batch['report']
            for prompt in prompts:
                img_path1 = finetuned_model.generate_and_save_imageV2(prompt)
                img_path2 = baseline_model.generate_and_save_imageV2(prompt)
                concat_images_with_prompt(img_path1, img_path2, prompt)
            break

        finetuned_scores = [test_loss, test_ssim, test_psnr]
        baseline_scores = [baseline_loss, baseline_ssim, baseline_psnr]
        self._plot_finetune_baseline_scores(finetuned_scores, baseline_scores)

    def save_best_model(self, path):
        temp_unet = copy.deepcopy(self.unet)
        temp_text_encoder = copy.deepcopy(self.text_encoder)

        merged_unet = temp_unet.merge_and_unload()
        merged_text_encoder = temp_text_encoder.merge_and_unload()

        merged_unet.save_pretrained(os.path.join(path, "unet"))
        merged_text_encoder.save_pretrained(os.path.join(path, "text_encoder"))

        del temp_unet, temp_text_encoder, merged_unet, merged_text_encoder
        torch.cuda.empty_cache()
        print(f"[INFO] Saved best LoRA model to: {path}")

    def _train_epoch(self, train_loader, optimizer, fold, epoch):
        self.unet.train()
        self.text_encoder.train()

        total_loss, num_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Training Fold {fold} - Epoch {epoch}"):
            images, texts = batch['image'], batch['report']
            images = images.to(self.device)
            loss = self._train_step(images, texts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=self.unet_lora_layers + self.text_encoder_lora_layers,
                max_norm=1.0
            )
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

                real_images = real_images.to(self.device)  # real_images has range [-1,1]
                real_images = (real_images + 1) / 2  # To range [0,1]
                prompts = [
                    f"{config.BASE_PROMPT_PREFIX}{text}{config.BASE_PROMPT_SUFFIX}" for text in texts
                ]
                generated_images = self.model.generate_images_Tensor(prompts)  # Tensor has range [0,1]
                if epoch % 2 == 0 and batch_idx % 4 == 0:
                    self._plot_image_pair(fold, epoch, batch_idx, real_images[:4], generated_images[:4])

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

                real_images = real_images.to(self.device)  # real_images has range [-1,1]
                real_images = (real_images + 1) / 2  # To range [0,1]
                prompts = [
                    f"{config.BASE_PROMPT_PREFIX}{text}{config.BASE_PROMPT_SUFFIX}" for text in texts
                ]
                generated_images = self.model.generate_images_Tensor(prompts)  # Tensor has range [0,1]
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
            f"{config.BASE_PROMPT_PREFIX}{text}{config.BASE_PROMPT_SUFFIX}" for text in texts
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

    def _plot_training_progressV2(self, train_losses, val_losses, ssim_scores, psnr_scores):
        import os
        import matplotlib.pyplot as plt

        # Train & Val Loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, "loss_curve.png"))
        plt.close()

        # SSIM & PSNR
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(ssim_scores, label="SSIM", color="green")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("SSIM")
        ax[0].set_title("Validation SSIM")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(psnr_scores, label="PSNR", color="blue")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("PSNR (dB)")
        ax[1].set_title("Validation PSNR")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, "ssim_psnr_curve.png"))
        plt.close()

    def _plot_image_pair(self, fold, epoch, batch_idx, real_images, gen_images):
        num_samples = real_images.size(0)
        figsize = (10, 2.5 * num_samples)
        fig, axes = plt.subplots(num_samples, 2, figsize=figsize, constrained_layout=True)

        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
        for i in range(num_samples):
            real_np = np.rot90(real_images[i, 0].cpu().numpy(), k=2)
            real_np = np.fliplr(real_np)
            gen_np = np.rot90(gen_images[i, 0].cpu().numpy(), k=2)
            gen_np = np.fliplr(gen_np)

            axes[i, 0].imshow(real_np, cmap='gray', origin='lower')
            if self.k_fold > 1:
                axes[i, 0].set_title(f"Fold {fold} Epoch {epoch} Batch {batch_idx} Sample {i} Real Image")
            else:
                axes[i, 0].set_title(f"Epoch {epoch} Batch {batch_idx} Sample {i} Real Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gen_np, cmap='gray', origin='lower')
            if self.k_fold > 1:
                axes[i, 1].set_title(f"Fold {fold} Epoch {epoch} Batch {batch_idx} Sample {i} Generated Image")
            else:
                axes[i, 1].set_title(f"Epoch {epoch} Batch {batch_idx} Sample {i} Generated Image")
            axes[i, 1].axis('off')

        plt.savefig(os.path.join(self.images_dir, f"fold{fold}_epoch{epoch}_batch{batch_idx}_comparison.png"), bbox_inches='tight')
        plt.close()

    def _plot_finetune_baseline_scores(self, finetuned_scores, baseline_scores):
        metrics = ['MSE Loss', 'SSIM', 'PSNR']
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width / 2, baseline_scores, width, label='Baseline')
        bars2 = ax.bar(x + width / 2, finetuned_scores, width, label='Fine-tuned')

        ax.bar_label(bars1, fmt='%.3f', padding=3)
        ax.bar_label(bars2, fmt='%.3f', padding=3)

        ax.set_ylabel('Scores')
        ax.set_title('Fine-tuned vs Baseline Model on Test Set')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, f"model_comparison.png"))
        plt.close()
