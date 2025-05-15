import optuna

from stable_diffusion_generator import XRayGenerator
from train import Trainer


def _objective(trial):
    # UNet LoRA
    r_unet = trial.suggest_int('r_unet', 8, 16)
    alpha_unet = trial.suggest_int('lora_alpha_unet', 10, 32)
    dropout_unet = trial.suggest_float('lora_dropout_unet', 0.001, 0.15)
    lr_unet = trial.suggest_float('lr_unet', 6e-5, 1.2e-4, log=True)
    wd_unet = trial.suggest_float('wd_unet', 0.1, 0.3)


    # Text Encoder LoRA
    r_text = trial.suggest_int('r_text', 10, 16)
    alpha_text = trial.suggest_int('lora_alpha_text', 20, 35)
    dropout_text = trial.suggest_float('lora_dropout_text', 0.01, 0.1)
    lr_text = trial.suggest_float('lr_text', 3e-6, 2e-5, log=True)
    wd_text = trial.suggest_float('wd_text', 0.005, 0.1)

    # scheduler
    T_max = trial.suggest_int('T_max', 3, 6)
    eta_min = trial.suggest_float('eta_min', 1e-5, 1e-4)


    model = XRayGenerator()
    trainer = Trainer(
        model=model,
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
        for_hpo=True,
        trial=trial
    )

    return trainer.train()


def run_hpo(n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials)

    print("Best trial:", study.best_trial)
    return study.best_trial.params
