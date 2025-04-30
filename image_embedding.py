import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel, Blip2Processor, Blip2Model, CLIPModel, CLIPImageProcessorFast

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


BATCH_SIZE = 4096
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = ['resnet50', 'efficientnet_b0', 'dinov2', 'clip']
NUM_WORKERS = 8

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def create_model_dirs(embedding_dir):
    for model_name in MODEL_NAMES:
        model_dir = os.path.join(embedding_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)


def get_preprocess(model_name):
    if model_name == 'resnet50':
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V2.transforms()

    elif model_name == 'efficientnet_b0':
        from torchvision.models import EfficientNet_B0_Weights
        return EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    elif model_name == 'clip':
        return CLIPImageProcessorFast.from_pretrained("openai/clip-vit-base-patch32")

    elif model_name == 'dinov2':
        return AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_hashes, image_dir, transform=None):
        self.image_hashes = image_hashes
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_hashes)

    def __getitem__(self, idx):
        img_hash = self.image_hashes[idx]
        img_path = os.path.join(self.image_dir, f"{img_hash}.jpg")

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_hash
        except Exception as e:
            return torch.zeros(3, 256, 256), img_hash


def load_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model.to(DEVICE)

    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()
        model.eval()
        return model.to(DEVICE)
    elif model_name == 'dinov2':
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        return model.to(DEVICE), processor
    elif model_name == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        processor = CLIPImageProcessorFast.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        model.eval()
        return model.to(DEVICE), processor
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_embeddings(image_hashes, model_name, image_dir, embedding_dir):
    preprocess = get_preprocess(model_name)
    model_input = load_model(model_name)

    if model_name in ['dinov2', 'clip']:
        model, processor = model_input
    else:
        model = model_input

    model_dir = os.path.join(embedding_dir, model_name)
    #
    # existing_files = set(f.split('.')[0] for f in os.listdir(model_dir) if f.endswith('.npy'))
    # hashes_to_process = [h for h in image_hashes if h not in existing_files]
    #
    # if not hashes_to_process:
    #     print(f"All embeddings for {model_name} already exist. Skipping.")
    #     return

    if model_name in ['resnet50', 'efficientnet_b0']:
        transform = preprocess
        use_processor = False
    else:
        transform = None
        use_processor = True

    dataset = ImageDataset(image_hashes, image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_images, batch_hashes in tqdm(dataloader, desc=f"Processing {model_name}"):
            if use_processor:
                inputs = preprocess(images=batch_images, return_tensors="pt").to(DEVICE)
            else:
                inputs = batch_images.to(DEVICE)

            if model_name == 'clip':
                features = model.get_image_features(**inputs)
            elif model_name in ['dinov2']:
                outputs = model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = model(inputs).squeeze(-1).squeeze(-1)

            features = features.cpu().numpy()
            for i, img_hash in enumerate(batch_hashes):
                if img_hash is not None:
                    np.save(os.path.join(model_dir, f"{img_hash}.npy"), features[i])


def main(input_df_folder, images_folder, output_folder):
    create_model_dirs(output_folder)

    all_dfs = []
    for file in os.listdir(input_df_folder):
        if file.endswith('.parquet'):
            file_path = os.path.join(input_df_folder, file)
            df = pd.read_parquet(file_path)
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"Combined DataFrame with {len(df)} rows")

    # Выводим все колонки
    print(df.columns)

    all_image_hashes = set(df['base_title_image'].dropna().unique()).union(
        set(df['cand_title_image'].dropna().unique()))
    all_image_hashes = [h for h in all_image_hashes if h is not None and isinstance(h, str) and h != '']

    print(f"Total unique images: {len(all_image_hashes)}")

    for model_name in MODEL_NAMES:
        print(f"\nProcessing {model_name}...")
        generate_embeddings(all_image_hashes, model_name, images_folder, output_folder)

    # df.to_parquet(os.path.join(output_folder, "combined_output.parquet"))


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Process images and parquet files to generate embeddings.")
    parser.add_argument("input_df_folder", type=str, help="Path to the folder containing the .parquet files")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing image files")
    parser.add_argument("output_folder", type=str, help="Path to the folder where output will be saved")

    args = parser.parse_args()

    main(args.input_df_folder, args.images_folder, args.output_folder)