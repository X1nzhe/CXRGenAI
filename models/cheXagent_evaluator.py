# import io
#
# import requests
# from typing import Optional, Union
#
# import torch
# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoTokenizer, CLIPModel, CLIPProcessor
# from PIL import Image
# from pathlib import Path
# from transformers.utils.logging import disable_progress_bar as transformers_disable_progress_bar
#
#
# class CheXagentEvaluator:
#     def __init__(self):
#         transformers_disable_progress_bar()
#         self.device = "cuda"
#         self.dtype = torch.bfloat16
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
#         self.model = self.model.to(self.dtype)
#         self.model.eval()
#
#     def evaluate_consistency(self, original_desc: str, image: Optional[Image.Image] = None,
#                              image_path: Optional[Union[str, Path]] = None):
#
#         if image is None and image_path is None:
#             raise ValueError("Either 'image' or 'image_path' must be provided.")
#         if not image:
#             image = Image.open(image_path)
#
#         prompt = (
#             f'Given the X-ray image(s), rate how well it matches the description: "{original_desc}". '
#             "Provide a score between 0 and 1, where 1 is a perfect match and 0 is completely unrelated."
#         )
#         # inputs = self.processor(
#         #     images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
#         # ).to(self.device, dtype=self.dtype)
#         #
#         # with torch.no_grad():
#         #     output = self.model.generate(**inputs, generation_config=self.generation_config)[0]
#         # response = self.processor.tokenizer.decode(output, skip_special_tokens=True)
#
#         query = self.tokenizer.from_list_format([{'image': image}, {'text': prompt}])
#         conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
#         input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
#         output = self.model.generate(
#             input_ids.to(self.device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
#             max_new_tokens=512
#         )[0]
#         response = self.tokenizer.decode(output[input_ids.size(1):-1])
#         print(response)
#
#         try:
#             score = float(response.strip())
#         except ValueError:
#             score = 0.0
#
#         return score
#
#
# # test
# if __name__ == "__main__":
#     # transformers_disable_progress_bar()
#     model_name = "microsoft/BiomedCLIP-PubMedBERT_256"
#     model = CLIPModel.from_pretrained(model_name)
#     processor = CLIPProcessor.from_pretrained(model_name)
