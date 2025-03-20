# import torch
# import torchvision.transforms as transforms
# from pytorch_msssim import ssim
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.text.bleu import BLEUScore
# from PIL import Image
#
#
# # SSIM (Structural Similarity Index Measure)
# def calculate_ssim(img1_path, img2_path):
#     img1 = Image.open(img1_path).convert("L")
#     img2 = Image.open(img2_path).convert("L")
#
#     transform = transforms.ToTensor()
#     img1 = transform(img1).unsqueeze(0)
#     img2 = transform(img2).unsqueeze(0)
#
#     ssim_score = ssim(img1, img2)
#     return ssim_score.item()
#
#
# # FID（Frechet Inception Distance）
# def calculate_fid(real_images, generated_images):
#     fid = FrechetInceptionDistance()
#     fid.update(real_images, real=True)
#     fid.update(generated_images, real=False)
#     return fid.compute().item()


# # BLEU (Bilingual Evaluation Understudy)
# def calculate_bleu(reference_texts, generated_texts):
#     bleu = BLEUScore(n_gram=4, smooth=True)
#     return bleu(reference_texts, generated_texts).item()