import io

import requests
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoTokenizer
from PIL import Image
from pathlib import Path
from transformers.utils.logging import disable_progress_bar as transformers_disable_progress_bar


class CheXagentEvaluator:
    def __init__(self, model_name="StanfordAIMI/CheXagent-8b", device="cuda" if torch.cuda.is_available() else "cpu"):
        transformers_disable_progress_bar()
        self.device = device
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)

    def evaluate_consistency(self, original_desc: str, image: Optional[Image.Image] = None,
                             image_path: Optional[Union[str, Path]] = None):
        if image is None and image_path is None:
            raise ValueError("Either 'image' or 'image_path' must be provided.")
        if not image:
            image = Image.open(image_path)

        prompt = (
            f'Given the X-ray image(s), rate how well it matches the description: "{original_desc}". '
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


# test
if __name__ == "__main__":
    transformers_disable_progress_bar()

    # device = "cuda"
    # dtype = torch.float16
    #
    # # step 2: Load Processor and Model
    # processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
    # generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
    # model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)
    #
    # # step 3: Fetch the images
    # image_path = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion" \
    #              "-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"
    # images = [Image.open(io.BytesIO(requests.get(image_path).content)).convert("RGB")]
    #
    # # step 4: Generate the Findings section
    # prompt = f'Describe "Airway"'
    # inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device,
    #                                                                                                      dtype=dtype)
    # output = model.generate(**inputs, generation_config=generation_config)[0]
    # response = processor.tokenizer.decode(output, skip_special_tokens=True)

    # step 1: Setup constant
    model_name = "StanfordAIMI/CheXagent-2-3b"
    dtype = torch.bfloat16
    device = "cuda"

    # step 2: Load Processor and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    model = model.to(dtype)
    model.eval()

    # step 3: Inference
    prompt = f'Describe "Airway"'
    paths = [
        "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"]
    query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
    conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    output = model.generate(
        input_ids.to(device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
        max_new_tokens=512
    )[0]
    response = tokenizer.decode(output[input_ids.size(1):-1])
