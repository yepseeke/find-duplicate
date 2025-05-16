import argparse

import torch
import os

from scipy.spatial.distance import chebyshev
from scipy.stats import pearsonr
from tqdm import tqdm

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_values = {
    "cosine_similarity": 0.0,
    "dot_product": 0.0,
    "euclidean_distance": 1e6,
    "manhattan_distance": 1e6,
    "chebyshev_distance": 1e6,
    "pearson_correlation": 0.0,
    "mean_absolute_difference": 1e6,
    "ratio_norms": 1.0,
    "projection_length": 0.0,
    "zero_crossings": 0
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None


def safe_load_tensor(image_path):
    if not os.path.exists(image_path):
        return None
    tensor = load_image(image_path)
    return tensor


def get_feature_extractor(model, model_name):
    if "resnet" in model_name:
        return torch.nn.Sequential(*list(model.children())[:-1])
    elif "vgg" in model_name:
        return model.features
    elif "densenet" in model_name:
        return torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_models():
    models_dict = {
        "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "vgg11": models.vgg11(weights=models.VGG11_Weights.DEFAULT),
        "densenet121": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    }

    for name in models_dict.keys():
        models_dict[name] = get_feature_extractor(models_dict[name].to(DEVICE).eval(), name)

    return models_dict


def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
        features = torch.flatten(features, 1)
    return features.cpu().numpy()


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))


def dot_product(vec1, vec2):
    return np.dot(vec1, vec2.T).item()


def feature_vector(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    dot = np.dot(vec1, vec2)
    eucl = np.linalg.norm(vec1 - vec2)
    manh = np.sum(np.abs(vec1 - vec2))
    cheb = chebyshev(vec1, vec2)
    corr, _ = pearsonr(vec1, vec2)
    mad = manh / len(vec1)
    ratio_norms = np.linalg.norm(vec1) / (np.linalg.norm(vec2) + 1e-10)
    proj_len = dot / (np.linalg.norm(vec2) + 1e-10)
    zero_crossings = np.sum(np.diff(np.sign(vec1 - vec2)) != 0)

    return {
        "cosine_similarity": cos_sim,
        "dot_product": dot,
        "euclidean_distance": eucl,
        "manhattan_distance": manh,
        "chebyshev_distance": cheb,
        "pearson_correlation": corr,
        "mean_absolute_difference": mad,
        "ratio_norms": ratio_norms,
        "projection_length": proj_len,
        "zero_crossings": zero_crossings,
    }


def process_dataframe(df, models_dict, image_dir):
    for model_name, model in models_dict.items():
        metrics = {
            "cosine_similarity": [],
            "dot_product": [],
            "euclidean_distance": [],
            "manhattan_distance": [],
            "chebyshev_distance": [],
            "pearson_correlation": [],
            "mean_absolute_difference": [],
            "ratio_norms": [],
            "projection_length": [],
            "zero_crossings": []
        }

        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Processing {model_name}"):
            batch = df.iloc[i:i + BATCH_SIZE]

            base_paths = [os.path.join(image_dir, f"{row.base_title_image}.jpg") for row in batch.itertuples()]
            cand_paths = [os.path.join(image_dir, f"{row.cand_title_image}.jpg") for row in batch.itertuples()]

            base_tensors = []
            cand_tensors = []
            valid_mask = []

            for b_path, c_path in zip(base_paths, cand_paths):
                b_tensor = safe_load_tensor(b_path)
                c_tensor = safe_load_tensor(c_path)
                if b_tensor is None or c_tensor is None:
                    valid_mask.append(False)
                    base_tensors.append(torch.zeros((3, 224, 224)))
                    cand_tensors.append(torch.zeros((3, 224, 224)))
                else:
                    valid_mask.append(True)
                    base_tensors.append(b_tensor.squeeze(0))
                    cand_tensors.append(c_tensor.squeeze(0))

            # Stack into batches
            base_batch = torch.stack(base_tensors).to(DEVICE)
            cand_batch = torch.stack(cand_tensors).to(DEVICE)

            # Extract features in batch
            base_vecs = extract_features(model, base_batch)  # modify extract_features to handle batches
            cand_vecs = extract_features(model, cand_batch)

            # Process each pair in batch
            for j, is_valid in enumerate(valid_mask):
                if not is_valid:
                    for key in metrics:
                        metrics[key].append(default_values[key])
                    continue

                feats = feature_vector(base_vecs[j], cand_vecs[j])
                for key, val in feats.items():
                    metrics[key].append(val)

        for key, vals in metrics.items():
            df[f"{model_name}-{key}"] = vals

    return df


def main():
    parser = argparse.ArgumentParser(description="Calculate image similarity features for a DataFrame.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to input .parquet DataFrame")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the updated DataFrame")

    args = parser.parse_args()

    df = pd.read_parquet(args.df_path)
    models_dict = load_models()

    df = process_dataframe(df, models_dict, args.image_dir)

    output_path = args.output_path or args.df_path.replace(".parquet", "_image_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved DataFrame with features to: {output_path}")


if __name__ == "__main__":
    main()