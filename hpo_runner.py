import optuna

from stable_diffusion_generator import XRayGenerator
from train import Trainer


def _objective(trial):
    # UNet LoRA
    r_unet = trial.suggest_int('r_unet', 4, 16)
    alpha_unet = trial.suggest_int('lora_alpha_unet', 8, 32)
    dropout_unet = trial.suggest_float('lora_dropout_unet', 0.0, 0.2)
    lr_unet = trial.suggest_float('lr_unet', 1e-5, 2e-4, log=True)
    wd_unet = trial.suggest_float('wd_unet', 0.0, 0.3)


    # Text Encoder LoRA
    r_text = trial.suggest_int('r_text', 4, 16)
    alpha_text = trial.suggest_int('lora_alpha_text', 8, 32)
    dropout_text = trial.suggest_float('lora_dropout_text', 0.0, 0.2)
    lr_text = trial.suggest_float('lr_text', 1e-6, 5e-5, log=True)
    wd_text = trial.suggest_float('wd_text', 0.0, 0.1)

    # scheduler
    T_max = trial.suggest_int('T_max', 3, 6)
    eta_min = trial.suggest_float('eta_min', 1e-6, 1e-4)

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])

    model = XRayGenerator()
    trainer = Trainer(
        model=model,
        batch_size=batch_size,
        epochs=2,
        unet_lora_config={
            'r': r_unet,
            'alpha': alpha_unet,
            'dropout': dropout_unet
        },
        text_lora_config={
            'r': r_text,
            'alpha': alpha_text,
            'dropout': dropout_text
        },
        scheduler_config={
            'T_max': T_max,
            'eta_min': eta_min
        },
        lr_unet=lr_unet,
        lr_text=lr_text,
        wd_unet=wd_unet,
        wd_text=wd_text,
        for_hpo=True
    )

    return trainer.train()


def run_hpo(n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials)
    study.optimize(_objective, timeout=1800)

    print("Best trial:", study.best_trial)
    return study.best_trial.params
