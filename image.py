import os
import sys
import torch

import pandas as pd
import numpy as np

from PIL import Image
from sentence_transformers import SentenceTransformer, util


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32', device=device)
device = torch.device(device)


def find_image_path(root_folder, image_hash):
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.startswith(image_hash):
                return os.path.join(subdir, file)
    return None


@torch.no_grad()
def encode_text(text):
    return model.encode(text, convert_to_tensor=True, use_fast=True)

@torch.no_grad()
def encode_image_safe(image_path):
    if image_path is None or not os.path.exists(image_path):
        return torch.tensor(np.zeros(512, dtype=np.float32), device=device)
    else:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        return model.encode(img, convert_to_tensor=True, use_fast=True)


def cosine_similarity(vec1, vec2):
    return util.cos_sim(vec1, vec2).item()


def compute_clip_features(df, image_root_folder):
    features = []

    for idx, row in df.iterrows():
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
        "title-title", "title-description", "title-image",
        "description-title", "description-description", "description-image",
        "image-title", "image-description", "image-image"
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