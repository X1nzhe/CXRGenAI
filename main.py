import argparse
import sys
import config

from stable_diffusion_generator import XRayGenerator
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Text-to-Medical-Image Model")
    parser.add_argument(
        "--env", choices=["dev", "product"], default="product", help="Choose runtime environment"
    )
    parser.add_argument(
        "--mode", choices=["train", "generate"], required=True, help="Choose mode：'train' or 'generate'"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to pre-trained model（'generate' mode only) "
    )
    parser.add_argument(
        "--description", type=str, help="Text description to generate X-Ray image（'generate' mode only) "
    )

    args = parser.parse_args()

    config.ENV = args.env
    config.reload_config()
    print(f"\nRunning with environment: {config.ENV}")

    if args.mode == "train":
        print("\nStart model training...")
        print(f"Epochs: {config.EPOCHS}, K_folds: {config.K_FOLDS}, Batch size: {config.BATCH_SIZE}, Image width: {config.IMAGE_WIDTH}, Image height: {config.IMAGE_HEIGHT}, Number of inference steps: {config.NUM_INFERENCE_STEPS}")

        model = XRayGenerator()
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
        print(f"Image width: {config.IMAGE_WIDTH}, Image height: {config.IMAGE_HEIGHT}, Number of inference steps: {config.NUM_INFERENCE_STEPS}")
        generator = XRayGenerator()
        try:
            print(f"Loading model from {args.model_path}")
            model = generator.load_model(args.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        prompt = [
            f"{config.BASE_PROMPT_PREFIX}{args.description}{config.BASE_PROMPT_SUFFIX}"
        ]
        generated_image_path = model.generate_and_save_image(prompt)
        print(f"Generated X-Ray image saved to path {generated_image_path}\n")


if __name__ == "__main__":
    main()
