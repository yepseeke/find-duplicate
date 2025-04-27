import os
import re
import sys
import torch

import pandas as pd
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device(device)

def remove_emojis(text):
    cleaned_text = re.sub(r'[^\w\s.,!?;:\'\"]+', '', text)
    return cleaned_text


def find_image_path(root_folder, image_hash):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith(image_hash) and file.endswith(".jpg"):
                    return os.path.join(folder_path, file)
    return None


@torch.no_grad()
def encode_text(text):
    text = text[0] if text is not None else ""

    text = remove_emojis(text)
    text = text.replace("\n", " ").replace('-', "")

    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    embeddings = clip_model.get_text_features(**inputs)
    return embeddings.squeeze(0)


@torch.no_grad()
def encode_image_safe(image_path):
    if image_path is None or not os.path.exists(image_path):
        return torch.tensor(np.zeros(512, dtype=np.float32), device=device)
    else:
        img = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
        return embeddings.squeeze(0)


def cosine_similarity(vec1, vec2):
    return util.cos_sim(vec1, vec2).item()


def compute_clip_features(df, image_root_folder):
    features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing CLIP features"):
        base_title_vec = encode_text([row['base_title']]).squeeze(0)
        base_description_vec = encode_text([row['base_description']]).squeeze(0)
        base_image_path = find_image_path(image_root_folder, row['base_title_image'])
        base_image_vec = encode_image_safe(base_image_path)

        cand_title_vec = encode_text([row['cand_title']]).squeeze(0)
        cand_description_vec = encode_text([row['cand_description']]).squeeze(0)
        cand_image_path = find_image_path(image_root_folder, row['cand_title_image'])
        cand_image_vec = encode_image_safe(cand_image_path)

        feature_vector = [
            cosine_similarity(base_title_vec, cand_title_vec),
            cosine_similarity(base_title_vec, cand_description_vec),
            cosine_similarity(base_title_vec, cand_image_vec),
            cosine_similarity(base_description_vec, cand_title_vec),
            cosine_similarity(base_description_vec, cand_description_vec),
            cosine_similarity(base_description_vec, cand_image_vec),
            cosine_similarity(base_image_vec, cand_title_vec),
            cosine_similarity(base_image_vec, cand_description_vec),
            cosine_similarity(base_image_vec, cand_image_vec),
        ]

        features.append(feature_vector)

    features_df = pd.DataFrame(features, columns=[
        "clip-title-title", "clip-title-description", "clip-title-image",
        "clip-description-title", "clip-description-description", "clip-description-image",
        "clip-image-title", "clip-image-description", "clip-image-image"
    ])
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    return df


def main(parquet_path, image_folder_path):
    print(f"Loading DataFrame from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print("Computing CLIP features...")
    df_with_features = compute_clip_features(df, image_folder_path)

    output_path = os.path.splitext(parquet_path)[0] + "_with_features.csv"
    print(f"Saving result to {output_path}")
    df_with_features.to_csv(output_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python image.py path_to_parquet path_to_image_folder")
    else:
        csv_path = sys.argv[1]
        image_folder_path = sys.argv[2]
        main(csv_path, image_folder_path)