import os
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader


class Trainer:
    def __init__(self, model, k_fold=5, batch_size=8, epochs=5, lr=1e-4, checkpoint_dir="../checkpoints"):
        self.model = model
        self.device = model.device

        self.unet = self.model.pipeline.unet
        self.text_encoder = self.model.pipeline.text_encoder
        self.tokenizer = self.model.pipeline.tokenizer
        self.vae = self.model.pipeline.vae
        self.noise_scheduler = self.model.pipeline.scheduler
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

        for fold_data in kfold_loaders:
            fold = fold_data['fold']
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']

            print(f"\nStarting training for Fold {fold}...")
            optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)
            self.unet.train()

            for epoch in range(self.epochs):
                train_loss = self._train_epoch(train_loader, optimizer, fold, epoch)
                val_loss, ssim, chexagent_score = self._validate_epoch(val_loader, ssim_metric, fold, epoch)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                ssim_scores.append(ssim)

                print(f"Fold {fold} - Epoch {epoch + 1} - Train Loss: {train_loss:.4f} "
                      f"- Val Loss: {val_loss:.4f} - SSIM Score: {ssim:.4f}")

            checkpoint_path = os.path.join(self.checkpoint_dir, f"sd_lora_fold{fold}.pth")
            self.model.save_model(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        self._plot_training_progress(train_losses, val_losses, ssim_scores)

    def _train_epoch(self, train_loader, optimizer, fold, epoch):
        total_loss, num_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Training Fold {fold} - Epoch {epoch + 1}"):
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
            for batch in tqdm(val_loader, desc=f"Validating Fold {fold} - Epoch {epoch + 1}"):
                real_images, texts = batch['image'], batch['report']
                real_images = real_images.to(self.device)
                generated_images = self.model.generate_images_for_ssim(texts)
                loss = self._train_step(real_images, texts)
                total_loss += loss.item()
                total_ssim += ssim_metric(generated_images, real_images).item()
                num_batches += 1
        return total_loss / max(num_batches, 1), total_ssim / max(num_batches, 1)

    def _train_step(self, images, texts):
        images = images.to(dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        text_embeddings = self.text_encoder(text_inputs.input_ids).last_hidden_state

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

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

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
