import os
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import time

from config import CHECKPOINTS_DIR, BASE_PROMPT
from data_loader import get_dataloader


def prepare_lora_model_for_training(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        modules_to_save=["conv_in"]
    )
    return get_peft_model(model, lora_config)


class Trainer:
    def __init__(self, model, k_fold=5, batch_size=8, epochs=5, lr=1e-4, checkpoint_dir=CHECKPOINTS_DIR):
        self.model = prepare_lora_model_for_training(model)
        self.device = model.device

        self.unet = model.unet
        self.text_encoder = model.text_encoder
        self.tokenizer = model.pipeline.tokenizer
        self.vae = model.vae
        self.noise_scheduler = model.pipeline.scheduler

        self.unet.enable_gradient_checkpointing()
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        kfold_loaders = get_dataloader(k_folds=self.k_fold, batch_size=self.batch_size)
        ssim_metric = SSIM(data_range=1.0).to(self.device)
        train_losses, val_losses, ssim_scores = [], [], []

        best_val_loss = float("inf")
        best_model_info = {"fold": None, "epoch": None, "path": None}

        # Record the start time
        start_time = time.time()
        print(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        for fold_data in kfold_loaders:
            fold = fold_data['fold']
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']

            print(f"\nStarting training for Fold {fold}...")
            optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)
            self.unet.train()

            for epoch in range(self.epochs):
                train_loss = self._train_epoch(train_loader, optimizer, fold, epoch)
                val_loss, ssim = self._validate_epoch(val_loader, ssim_metric, fold, epoch)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                ssim_scores.append(ssim)

                print(f"Fold {fold} - Epoch {epoch} - Train Loss: {train_loss:.4f} "
                      f"- Val Loss: {val_loss:.4f} - SSIM Score: {ssim:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_info["fold"] = fold
                    best_model_info["epoch"] = epoch
                    best_model_info["path"] = os.path.join(self.checkpoint_dir, f"best_model.pth")

                    self.model.save_model(best_model_info["path"])
                    print(
                        f"Best model updated: Fold {fold}, Epoch {epoch}, "
                        f"Val Loss {val_loss:.4f}, saved to {best_model_info['path']}")
            # checkpoint_path = os.path.join(self.checkpoint_dir, f"sd_lora_fold{fold}.pth")
            # self.model.save_model(checkpoint_path)
            # print(f"Checkpoint saved: {checkpoint_path}")

        # After training loop
        # Record the end time
        end_time = time.time()
        total_time = end_time - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds):.2f} seconds")
        self._plot_training_progress(train_losses, val_losses, ssim_scores)

        print("\nTraining complete. Running final test on the best model from {self.best_model_info['path']}...\n")
        self.model.load_model(best_model_info["path"])
        test_loader = kfold_loaders[0]['test_loader']
        test_loss, test_ssim = self._test_epoch(test_loader, ssim_metric)
        print(f"Final Test - Loss: {test_loss:.4f}, SSIM: {test_ssim:.4f}")
        print(f"Generating some images...")
        for batch in test_loader:
            prompt = batch['report']
            self.model.generate_and_save_image(prompt)
            break

    def _train_epoch(self, train_loader, optimizer, fold, epoch):
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

    def _validate_epoch(self, val_loader, ssim_metric, fold, epoch):
        total_loss, total_ssim, num_batches = 0.0, 0.0, 0
        self.model.pipeline.unet.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Fold {fold} - Epoch {epoch}"):
                real_images, texts = batch['image'], batch['report']

                real_images = real_images.to(self.device)
                if real_images.dtype == torch.uint8:  # normalize to the range [0, 1]
                    real_images = real_images.float() / 255.0
                prompts = [
                    f"{BASE_PROMPT} Diagnosis: {t}" for t in texts
                ]
                generated_images = self.model.generate_images_for_ssimV2(prompts)  # test new method
                loss = self._compute_test_loss(generated_images, real_images)

                total_loss += loss.item()
                total_ssim += ssim_metric(generated_images, real_images).item()
                num_batches += 1
        return total_loss / max(num_batches, 1), total_ssim / max(num_batches, 1)

    def _test_epoch(self, test_loader, ssim_metric):
        total_loss, total_ssim, num_batches = 0.0, 0.0, 0
        self.model.pipeline.unet.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing"):
                real_images, texts = batch['image'], batch['report']

                real_images = real_images.to(self.device)
                if real_images.dtype == torch.uint8: # normalize to the range [0, 1]
                    real_images = real_images.float() / 255.0
                prompts = [
                    f"{BASE_PROMPT}{text}" for text in texts
                ]
                generated_images = self.model.generate_images_for_ssimV2(prompts)  # test new method
                loss = self._compute_test_loss(generated_images, real_images)

                total_loss += loss.item()
                total_ssim += ssim_metric(generated_images, real_images).item()
                num_batches += 1
        return total_loss / max(num_batches, 1), total_ssim / max(num_batches, 1)

    def _compute_test_loss(self, generated_images, real_images):
        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(generated_images, real_images)
            l1_loss = torch.nn.functional.l1_loss(generated_images, real_images)
            loss = 0.8 * mse_loss + 0.2 * l1_loss
        return loss

    def _train_step(self, images, texts):
        images = images.to(dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        prompts = [
            f"{BASE_PROMPT}{t}" for t in texts
        ]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        text_embeddings = self.text_encoder(input_ids=text_inputs.input_ids.to(self.device),
                                            attention_mask=text_inputs.attention_mask.to(self.device)
                                            ).last_hidden_state

        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
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

        mse_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        l1_loss = torch.nn.functional.l1_loss(noise_pred, noise)
        loss = 0.8 * mse_loss + 0.2 * l1_loss
        return loss

    def _plot_training_progress(self, train_losses, val_losses, ssim_scores):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(ssim_scores, label="SSIM Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.legend()
        plt.title("SSIM Score")

        plt.savefig(os.path.join(self.checkpoint_dir, "training_progress.png"))
        plt.show()
