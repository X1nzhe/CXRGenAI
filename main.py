import argparse
import sys


from config import BASE_PROMPT_PREFIX, BASE_PROMPT_SUFFIX
from models.stable_diffusion_generator import XRayGenerator
from train import Trainer
# from evaluate import Evaluator
# from generate import Generator
# from models.cheXagent_evaluator import CheXagentEvaluator


def main():
    parser = argparse.ArgumentParser(description="Text-to-Medical-Image Model")
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
    if args.mode == "train":
        print("Start model training...")
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
        generator = XRayGenerator()
        try:
            print(f"Loading model from {args.model_path}")
            # model.load_model(args.model_path)
            model = generator.load_model(args.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        prompt = [
            f"{BASE_PROMPT_PREFIX}{args.description}{BASE_PROMPT_SUFFIX}"
        ]
        generated_image_path = model.generate_and_save_image(prompt)
        print(f"Generated X-Ray image saved to path {generated_image_path}\n")

        
        # print("Using CheXagent to evaluate the generated X-Ray image...")
        # evaluator = CheXagentEvaluator()
        # score = evaluator.evaluate_consistency(original_desc=args.description, image_path=generated_image_path,)
        # print(f"Generated X-Ray image consistency score：{score}")


if __name__ == "__main__":
    main()
