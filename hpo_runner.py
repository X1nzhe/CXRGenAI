import optuna

from stable_diffusion_generator import XRayGenerator
from train import Trainer


def _objective(trial):
    # UNet LoRA
    r_unet = trial.suggest_int('r_unet', 2, 16)
    alpha_unet = trial.suggest_int('lora_alpha_unet', 2, 64)
    dropout_unet = trial.suggest_float('lora_dropout_unet', 0.0, 0.3)

    # Text Encoder LoRA
    r_text = trial.suggest_int('r_text', 2, 16)
    alpha_text = trial.suggest_int('lora_alpha_text', 2, 64)
    dropout_text = trial.suggest_float('lora_dropout_text', 0.0, 0.3)

    # scheduler
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    T_max = trial.suggest_int('T_max', 2, 10)
    eta_min = trial.suggest_float('eta_min', lr * 0.001, lr * 0.1)

    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])

    model = XRayGenerator()
    trainer = Trainer(
        model=model,
        lr=lr,
        batch_size=batch_size,
        epochs=10,
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
        for_hpo=True
    )

    return trainer.train()


def run_hpo(n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials)

    print("Best trial:", study.best_trial)
    return study.best_trial.params
