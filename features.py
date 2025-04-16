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
