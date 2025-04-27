import os
import re
import sys
import torch

import pandas as pd
import numpy as np

from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel, CLIPImageProcessor, CLIPImageProcessorFast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_image_processor = CLIPImageProcessorFast.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
device = torch.device(device)

BATCH_SIZE = 12


def build_image_index(root_folder):
    image_index = {}
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_hash = file.split('.')[0]
                image_index[image_hash] = os.path.join(subdir, file)
    return image_index


def find_image_path(image_index, image_hash):
    return image_index.get(image_hash, None)


def remove_emojis(text):
    cleaned_text = re.sub(r'[^\w\s.,!?;:\'\"]+', '', text)
    return cleaned_text


@torch.no_grad()
def encode_text_batch(texts):
    cleaned_texts = [remove_emojis(t).replace("\n", " ").replace('-', "") if t is not None else "" for t in texts]
    inputs = clip_text_processor(text=cleaned_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    embeddings = clip_model.get_text_features(**inputs)
    return embeddings


@torch.no_grad()
def encode_image_safe(image_path):
    if image_path is None or not os.path.exists(image_path):
        return torch.tensor(np.zeros(512, dtype=np.float32), device=device)

    try:
        img = Image.open(image_path).convert('RGB')
    except (UnidentifiedImageError, OSError):
        return torch.tensor(np.zeros(512, dtype=np.float32), device=device)

    inputs = clip_image_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.squeeze(0)


def batch_cosine_sim(a, b):
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return (a @ b.T).diagonal()


def compute_clip_features(df, image_root_folder):
    features = []
    image_index = build_image_index(image_root_folder)

    for batch_start in tqdm(range(0, len(df), BATCH_SIZE), desc="Computing CLIP features"):
        batch_df = df.iloc[batch_start:batch_start + BATCH_SIZE]

        base_titles = batch_df['base_title'].tolist()
        base_descriptions = batch_df['base_description'].tolist()
        cand_titles = batch_df['cand_title'].tolist()
        cand_descriptions = batch_df['cand_description'].tolist()

        base_image_vecs = []
        cand_image_vecs = []
        for idx, row in batch_df.iterrows():
            base_image_path = find_image_path(image_index, row['base_title_image'])
            base_image_vecs.append(encode_image_safe(base_image_path))

            cand_image_path = find_image_path(image_index, row['cand_title_image'])
            cand_image_vecs.append(encode_image_safe(cand_image_path))

        base_image_vecs = torch.stack(base_image_vecs)
        cand_image_vecs = torch.stack(cand_image_vecs)

        base_title_vecs = encode_text_batch(base_titles)
        base_description_vecs = encode_text_batch(base_descriptions)
        cand_title_vecs = encode_text_batch(cand_titles)
        cand_description_vecs = encode_text_batch(cand_descriptions)

        title_title_sim = batch_cosine_sim(base_title_vecs, cand_title_vecs)
        title_desc_sim = batch_cosine_sim(base_title_vecs, cand_description_vecs)
        title_image_sim = batch_cosine_sim(base_title_vecs, cand_image_vecs)

        desc_title_sim = batch_cosine_sim(base_description_vecs, cand_title_vecs)
        desc_desc_sim = batch_cosine_sim(base_description_vecs, cand_description_vecs)
        desc_image_sim = batch_cosine_sim(base_description_vecs, cand_image_vecs)

        image_title_sim = batch_cosine_sim(base_image_vecs, cand_title_vecs)
        image_desc_sim = batch_cosine_sim(base_image_vecs, cand_description_vecs)
        image_image_sim = batch_cosine_sim(base_image_vecs, cand_image_vecs)

        batch_features = torch.stack([
            title_title_sim,
            title_desc_sim,
            title_image_sim,
            desc_title_sim,
            desc_desc_sim,
            desc_image_sim,
            image_title_sim,
            image_desc_sim,
            image_image_sim
        ], dim=1).cpu().numpy()

        features.extend(batch_features)

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