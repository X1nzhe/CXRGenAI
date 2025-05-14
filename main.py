import argparse
import sys
import config

from stable_diffusion_generator import XRayGenerator
from train import Trainer
from hpo_runner import run_hpo


def main():
    parser = argparse.ArgumentParser(description="Text-to-Medical-Image Model")
    parser.add_argument(
        "--env", choices=["dev", "product"], default="product", help="Choose runtime environment"
    )
    parser.add_argument(
        "--mode", choices=["train", "generate"], help="Choose mode：'train' or 'generate'"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to pre-trained model（'generate' mode only) "
    )
    parser.add_argument(
        "--description", type=str, help="Text description to generate X-Ray image（'generate' mode only) "
    )
    parser.add_argument(
        "--hpo", default=False, help="Enable hyperparameter optimization. Can be used with or without 'Train' mode."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10). Can be used independently or together with 'Train' mode."
    )

    args = parser.parse_args()
    if args.mode is None and not args.hpo:
        parser.error("You must specify either --mode or --hpo (or both).")

    config.ENV = args.env
    config.reload_config()
    print(f"\nRunning with environment: {config.ENV}")

    best_params = None
    if args.hpo:
        print("\n Hyperparameters searching...")
        best_params = run_hpo(n_trials=args.n_trials)

    if args.mode == "train":
        print("\nStart model training...")
        print(f"Epochs: {config.EPOCHS}, K_folds: {config.K_FOLDS}, Batch size: {config.BATCH_SIZE}, Image width: {config.IMAGE_WIDTH}, Image height: {config.IMAGE_HEIGHT}, Number of inference steps: {config.NUM_INFERENCE_STEPS}")
        model = XRayGenerator()
        if best_params:
            trainer = Trainer(model, unet_lora_config=best_params.unet_lora_config,
                              text_lora_config=best_params.text_lora_config, scheduler_config=best_params.scheduler_config,
                              lr_unet=best_params.lr_unet, lr_text=best_params.lr_text,
                              wd_unet=best_params.wd_unet, wd_text=best_params.wd_text)
        else:
            trainer = Trainer(model)

        trainer.train()

    elif args.mode == "generate":
        if not args.description:
            print("Error：Please provide text description for the generated X-Ray image ")
            sys.exit(1)
        if not args.model_path:
            print("Error：Please provide a path to the GenAI model for the generated X-Ray image ")
            sys.exit(1)

        print("\nStart X-Ray image generating...")
        print(
            f"Image width: {config.IMAGE_WIDTH}, Image height: {config.IMAGE_HEIGHT}, Number of inference steps: {config.NUM_INFERENCE_STEPS}")
        generator = XRayGenerator()
        try:
            print(f"Loading model from {args.model_path}")
            model = generator.load_modelV2(args.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        prompt = [
            f"{config.BASE_PROMPT_PREFIX}{args.description}{config.BASE_PROMPT_SUFFIX}"
        ]
        generated_image_path = model.generate_and_save_imageV2(prompt)
        print(f"Generated X-Ray image saved to path {generated_image_path}\n")


if __name__ == "__main__":
    main()
