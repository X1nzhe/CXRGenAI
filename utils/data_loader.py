import os
import requests
import tarfile
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from PIL import Image
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Indiana University Chest X-ray Collection dataset
PNGS_FILENAME = "NLMCXR_png.tgz"
REPORTS_FILENAME = "NLMCXR_reports.tgz"
DATASET_URL = f"https://openi.nlm.nih.gov/imgs/collections/{PNGS_FILENAME}"
REPORTS_URL = f"https://openi.nlm.nih.gov/imgs/collections/{REPORTS_FILENAME}"
DATA_DIR = "../data"

# Default number of threads for data downloading
NUM_THREADS = 8


def download_chunk(url, start, end, dest_path, progress_bar):
    headers = {"Range": f"bytes={start}-{end}"}
    response = requests.get(url, headers=headers, stream=True)

    with open(dest_path, "r+b") as file:
        file.seek(start)
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_file_multithread(url, dest_path, num_threads=NUM_THREADS):
    response = requests.head(url)
    total_size = int(response.headers.get("content-length", 0))

    if total_size == 0:
        raise ValueError("Unable to fetch the file")

    chunk_size = total_size // num_threads

    with open(dest_path, "wb") as file:
        file.truncate(total_size)

    progress_bar = tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        unit="B",
        unit_scale=True
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_threads - 1 else total_size - 1
            futures.append(executor.submit(download_chunk, url, start, end, dest_path, progress_bar))

        for future in futures:
            future.result()

    progress_bar.close()
    print("Download completed:", dest_path)


def download_file_singlethread(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024

    with open(dest_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(dest_path)}",
            total=total_size,
            unit="B",
            unit_scale=True
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def extract_tarfile(file_path, extract_dir):
    with tarfile.open(file_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(file_path)}"):
            tar.extract(member, path=extract_dir)


def parse_reports(reports_dir):
    report_data = []
    reports_dir = f"{reports_dir}/ecgen-radiology"

    for filename in tqdm(os.listdir(reports_dir), desc="Parsing reports"):
        if filename.endswith('.xml'):
            xml_path = os.path.join(reports_dir, filename)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                image_ids = []
                for image_node in root.findall(".//parentImage"):
                    image_id = image_node.get("id")
                    if image_id:
                        image_ids.append(image_id)

                findings = ""
                impression = ""

                for text_node in root.findall(".//AbstractText"):
                    label = text_node.get("Label")
                    if label == "FINDINGS":
                        findings = text_node.text or ""
                    elif label == "IMPRESSION":
                        impression = text_node.text or ""

                full_report = findings + " " + impression
                full_report = re.sub(r'\s+', ' ', full_report).strip()

                for img_id in image_ids:
                    report_data.append({
                        'image_id': img_id,
                        'report': full_report
                    })

            except ET.ParseError:
                print(f"Error parsing {xml_path}")
                continue

    return pd.DataFrame(report_data)


def check_and_download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    png_dir = os.path.join(DATA_DIR, "png")
    png_tgz_path = os.path.join(DATA_DIR, PNGS_FILENAME)

    print("Checking for Indiana University Chest X-ray Collection dataset...")
    if not os.path.exists(png_dir) or len(os.listdir(png_dir)) == 0:
        if os.path.exists(PNGS_FILENAME):
            print("PNG zipfile exists.")
        else:
            print(f"Downloading PNG images from {DATASET_URL}")
            download_file_multithread(DATASET_URL, png_tgz_path)

        print("Extracting PNG images...")
        os.makedirs(png_dir, exist_ok=True)
        extract_tarfile(png_tgz_path, png_dir)

        print("PNG images extracted.")
    else:
        print("PNG images are already available.")

    reports_dir = os.path.join(DATA_DIR, "reports")
    reports_tgz_path = os.path.join(DATA_DIR, REPORTS_FILENAME)

    if not os.path.exists(reports_dir) or len(os.listdir(reports_dir)) == 0:
        if os.path.exists(PNGS_FILENAME):
            print("Reports zipfile exists.")
        else:
            print(f"Downloading reports from {REPORTS_URL}")
            download_file_singlethread(REPORTS_URL, reports_tgz_path)

        print("Extracting reports...")
        os.makedirs(reports_dir, exist_ok=True)
        extract_tarfile(reports_tgz_path, reports_dir)

        print("Reports extracted.")
    else:
        print("Reports are already available.")

    csv_path = os.path.join(DATA_DIR, "metadata.csv")
    if not os.path.exists(csv_path):
        print("Processing reports and creating metadata CSV...")
        reports_df = parse_reports(reports_dir)

        valid_entries = []
        for _, row in tqdm(reports_df.iterrows(), desc="Validating images", total=len(reports_df)):
            img_id = row['image_id']
            img_path = os.path.join(png_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                valid_entries.append({
                    'image_filename': f"{img_id}.png",
                    'report': row['report']
                })

        final_df = pd.DataFrame(valid_entries)
        final_df.to_csv(csv_path, index=False)
        print(f"Created metadata CSV with {len(final_df)} valid entries.")
    else:
        print("Metadata CSV file already exists.")


class XRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_length=256):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        report = self.data_frame.iloc[idx, 1]

        try:
            image = Image.open(img_name).convert("RGB")

            if self.transform:
                image = self.transform(image)

            if len(report) > self.max_length:
                report = report[:self.max_length]

            return {
                'image': image,
                'report': report,
                'image_path': img_name
            }

        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            dummy_image = torch.zeros((3, 256, 256)) if self.transform else Image.new('RGB', (256, 256))
            return {
                'image': dummy_image,
                'report': "Error loading image",
                'image_path': img_name
            }


def get_dataloader(k_folds=5, batch_size=8, random_seed=123):
    check_and_download_dataset()
    torch.manual_seed(random_seed)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_dataset = XRayDataset(
        csv_file=os.path.join(DATA_DIR, "metadata.csv"),
        img_dir=os.path.join(DATA_DIR, "png"),
        transform=transform
    )

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    kfold_loaders = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold}')
        print(f'Train: {len(train_ids)} | Validation: {len(val_ids)}')

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        kfold_loaders.append({
            'fold': fold,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_size': len(train_ids),
            'val_size': len(val_ids)
        })

    return kfold_loaders


# # test
# if __name__ == "__main__":
#
#     kfold_loaders = get_dataloader()
#
#     sample_fold = kfold_loaders[0]
#     print(f"Fold: {sample_fold['fold']}")
#     print(f"Training samples: {sample_fold['train_size']}")
#     print(f"Validation samples: {sample_fold['val_size']}")
#
#     train_loader = sample_fold['train_loader']
#     for batch in train_loader:
#         print(f"Batch image shape: {batch['image'].shape}")
#         print(f"Sample report: {batch['report'][0][:100]}...")
#         break
