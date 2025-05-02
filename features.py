import time
import os
import nltk
import torch
import gc

from retsim_pytorch import RETSimConfig, RETSim
from safetensors.torch import load_file as load_safetensors
from retsim_pytorch.preprocessing import binarize
from tqdm import tqdm
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from Levenshtein import distance as levenshtein_distance
from torch.nn.functional import cosine_similarity
import pandas as pd
import numpy as np

nltk.download('punkt')


def jaccard_similarity(text1, text2):
    tokens1 = set(nltk.word_tokenize(text1.lower()))
    tokens2 = set(nltk.word_tokenize(text2.lower()))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0


def cosine_distance(df, column_a, column_b, model_name, batch_size=128):
    start = time.time()
    model = SentenceTransformer(model_name)
    print(f"Модель загружена за {time.time() - start:.2f} секунд.\n")

    bert_scores = []
    print(f"Вычисляется косинусное сходство между {column_a, column_b}...")
    for i in tqdm(range(0, len(df), batch_size), desc="Обработка батчей"):
        batch = df.iloc[i:i + batch_size].copy()

        base_embeddings = model.encode(batch[column_a].tolist(), convert_to_tensor=True, show_progress_bar=False)
        cand_embeddings = model.encode(batch[column_b].tolist(), convert_to_tensor=True, show_progress_bar=False)

        sim_matrix = util.cos_sim(base_embeddings, cand_embeddings)

        scores = [sim_matrix[j][j].item() for j in range(len(batch))]
        bert_scores.extend(scores)

        del base_embeddings, cand_embeddings, sim_matrix
        torch.cuda.empty_cache()
        gc.collect()

    df[model_name] = bert_scores


def pad_text_pairs_to_equal_length(texts_a: List[str], texts_b: List[str], pad_char: str = " ") \
        -> Tuple[List[str], List[str]]:
    padded_a = []
    padded_b = []

    for a, b in zip(texts_a, texts_b):
        len_a = len(a)
        len_b = len(b)
        max_len = max(len_a, len_b)

        padded_a.append(a.ljust(max_len, pad_char))
        padded_b.append(b.ljust(max_len, pad_char))

    return padded_a, padded_b


def cosine_distance_retsim(df, column_a, column_b, model, device='cpu', batch_size=128):
    model.to(device)
    model.eval()

    all_scores = []
    print(f"Вычисляется косинусное сходство между '{column_a}' и '{column_b}'...")

    for i in tqdm(range(0, len(df), batch_size), desc="Обработка батчей"):
        batch = df.iloc[i:i + batch_size].copy()
        texts_a = batch[column_a].tolist()
        texts_b = batch[column_b].tolist()

        padded_texts_a, padded_texts_b = pad_text_pairs_to_equal_length(texts_a, texts_b)

        bin_inputs_a, _ = binarize(padded_texts_a)
        bin_inputs_b, _ = binarize(padded_texts_b)


        tensor_a = torch.tensor(bin_inputs_a).to(device)
        tensor_b = torch.tensor(bin_inputs_b).to(device)

        with torch.no_grad():
            emb_a, _ = model(tensor_a)
            emb_b, _ = model(tensor_b)

        sim = cosine_similarity(emb_a, emb_b).cpu().tolist()
        all_scores.extend(sim)

        del tensor_a, tensor_b, emb_a, emb_b, sim
        torch.cuda.empty_cache()
        gc.collect()



    df['retsim_score'] = all_scores


def apply_lavenstein(df):
    df['levenshtein_title'] = df.apply(
        lambda row: levenshtein_distance(row['base_title'], row['cand_title']) / max(len(row['base_title']), 1),
        axis=1
    )


def apply_jaccard(df):
    df['jaccard_title'] = df.apply(lambda row: jaccard_similarity(row['base_title'], row['cand_title']), axis=1)


def load_model_weights(model: torch.nn.Module, path: str, safetensors: bool = True) -> None:
    if safetensors:
        print(f"Загрузка SafeTensors весов из: {path}")
        state_dict = load_safetensors(path)
    else:
        print(f"Загрузка стандартных PyTorch весов из: {path}")
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)


def build_vector_features(df: pd.DataFrame, vector_folder: str, output_path: str, model_name: str) -> pd.DataFrame:
    features = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_hash = row['base_image_title']
        cand_hash = row['cand_image_title']

        base_path = os.path.join(vector_folder, f"{base_hash}.npy")
        cand_path = os.path.join(vector_folder, f"{cand_hash}.npy")

        if os.path.exists(base_path) and os.path.exists(cand_path):
            base_vec = np.load(base_path)
            cand_vec = np.load(cand_path)

            cosine_sim = cosine_similarity([base_vec], [cand_vec])[0][0]
            l2_dist = euclidean(base_vec, cand_vec)
            manhattan_dist = cityblock(base_vec, cand_vec)
            dot_product = np.dot(base_vec, cand_vec)
            abs_diff_sum = np.sum(np.abs(base_vec - cand_vec))
        else:
            cosine_sim = 0.0
            l2_dist = 0.0
            manhattan_dist = 0.0
            dot_product = 0.0
            abs_diff_sum = 0.0

        features.append({
            f'{model_name}_cosine_similarity': cosine_sim,
            f'{model_name}_l2_distance': l2_dist,
            f'{model_name}_manhattan_distance': manhattan_dist,
            f'{model_name}_dot_product': dot_product,
            f'{model_name}_abs_diff_sum': abs_diff_sum
        })

    features_df = pd.DataFrame(features)
    features_df.to_parquet(output_path, index=False)
    return features_df


def features_calculation(df):
    df['is_same_param1'] = (df['base_param1'] == df['cand_param1']).astype(int)
    df['is_same_param2'] = (df['base_param2'] == df['cand_param2']).astype(int)
    df['images_diff'] = df['base_count_images'] - df['cand_count_images']
    df['base_json_params_parsed'] = df['base_json_params'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    df['cand_json_params_parsed'] = df['cand_json_params'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    df['common_keys_count'] = df.apply(common_keys, axis=1)
    df['common_values_count'] = df.apply(common_values, axis=1)
    df['common_keys_ratio'] = df.apply(common_keys_ratio, axis=1)
    df['is_same_category'] = (df['base_category_name'] == df['cand_category_name']).astype(int)
    df['is_same_images_count'] = (df['base_count_images'] == df['cand_count_images']).astype(int)
    df['title_exact_match'] = (df['base_title'] == df['cand_title']).astype(int)
    df['title_jaccard'] = df.apply(lambda row: jaccard_similarity(row['base_title'], row['cand_title']), axis=1)
    df['price_diff'] = df['base_price'] - df['cand_price']
    df['price_ratio'] = df['base_price'] / df['cand_price'].replace(0, float('inf'))
    df['base_description_len'] = df['base_description'].str.len()
    df['cand_description_len'] = df['cand_description'].str.len()
    df['description_len_diff'] = df['base_description_len'] - df['cand_description_len']
    df['is_same_title_image'] = (df['base_title_image'] == df['cand_title_image']).astype(int)

    df.to_parquet(output_path, index=False)
    return df


def basic_text_features(df, text_col1, text_col2):
    df[f'{text_col1}_len'] = df[text_col1].str.len()
    df[f'{text_col2}_len'] = df[text_col2].str.len()
    df[f'{text_col1}_{text_col2}_len_diff'] = abs(df[f'{text_col1}_len'] - df[f'{text_col2}_len'])
    df[f'{text_col1}_ru_chars'] = df[text_col1].apply(lambda x: sum(c.isalpha() and ord(c) > 127 for c in str(x)))
    df[f'{text_col2}_ru_chars'] = df[text_col2].apply(lambda x: sum(c.isalpha() and ord(c) > 127 for c in str(x)))
    df[f'{text_col1}_en_chars'] = df[text_col1].apply(lambda x: sum(c.isalpha() and ord(c) < 128 for c in str(x)))
    df[f'{text_col2}_en_chars'] = df[text_col2].apply(lambda x: sum(c.isalpha() and ord(c) < 128 for c in str(x)))
    df[f'{text_col1}_digits'] = df[text_col1].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df[f'{text_col2}_digits'] = df[text_col2].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df[f'{text_col1}_unique_chars'] = df[text_col1].apply(lambda x: len(set(str(x))))
    df[f'{text_col2}_unique_chars'] = df[text_col2].apply(lambda x: len(set(str(x))))

    df.to_parquet(output_path, index=False)
    return df


if __name__ == '__main__':
    df = pd.read_parquet('train_4.parquet')

    config = RETSimConfig()
    model = RETSim(config)

    load_model_weights(model, 'weights/model.safetensors')

    cosine_distance_retsim(df, 'base_description', 'cand_description', model, device='cuda')
    cosine_distance(df, 'base_title', 'cand_title', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    df = df.rename(columns={'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': 'mpnet-base-v2'})

    # apply_jaccard(df)
    # apply_lavenstein(df)

    output_path = 'train_4.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n✅ Результаты сохранены: {os.path.abspath(output_path)}")
