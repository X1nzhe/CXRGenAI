import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image


class CheXagentEvaluator:
    def __init__(self, model_name="StanfordAIMI/CheXagent-8b", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)

    def evaluate_consistency(self, image_path, original_text):
        image = Image.open(image_path).convert("L")
        prompt = (
            f'Given this X-ray, rate how well it matches the description: "{original_text}". '
            "Provide a score between 0 and 1, where 1 is a perfect match and 0 is completely unrelated."
        )
        inputs = self.processor(
            images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            output = self.model.generate(**inputs, generation_config=self.generation_config)[0]
        response = self.processor.tokenizer.decode(output, skip_special_tokens=True)

        try:
            score = float(response.strip())
        except ValueError:
            score = 0.0

        return score
