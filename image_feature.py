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


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


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
        "efficientnet_b0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "densenet121": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    }

    for name in models_dict.keys():
        models_dict[name] = get_feature_extractor(models_dict[name].to(DEVICE).eval(), name)

    return models_dict


def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
        features = torch.flatten(features, 1)
    return features.squeeze().cpu().numpy()


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

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model_name}"):
            base_path = os.path.join(image_dir, f"{row['base_image_title']}.jpg")
            cand_path = os.path.join(image_dir, f"{row['cand_image_title']}.jpg")

            if not os.path.exists(base_path) or not os.path.exists(cand_path):
                metrics["cosine_similarity"].append(0.0)
                metrics["dot_product"].append(0.0)
                metrics["euclidean_distance"].append(1e6)
                metrics["manhattan_distance"].append(1e6)
                metrics["chebyshev_distance"].append(1e6)
                metrics["pearson_correlation"].append(0.0)
                metrics["mean_absolute_difference"].append(1e6)
                metrics["ratio_norms"].append(1.0)
                metrics["projection_length"].append(0.0)
                metrics["zero_crossings"].append(0)
                continue

            base_tensor = load_image(base_path)
            cand_tensor = load_image(cand_path)

            vec1 = extract_features(model, base_tensor).reshape(-1)
            vec2 = extract_features(model, cand_tensor).reshape(-1)

            feats = feature_vector(vec1, vec2)
            if feats is None:
                for key in metrics.keys():
                    metrics[key].append(np.nan)
                continue

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

    output_path = args.output_path or args.df_path.replace(".parquet", "_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved DataFrame with features to: {output_path}")


if __name__ == "__main__":
    main()