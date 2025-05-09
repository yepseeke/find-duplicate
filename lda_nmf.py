import os
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import pymorphy2
from gensim import corpora, models
from scipy.spatial.distance import cosine, jensenshannon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# df = pd.read_csv('your_data.csv')

stop_words_ru = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

def preprocess_ru(text):
    tokens = word_tokenize(text.lower(), language='russian')
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words_ru]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return lemmas


def add_topic_similarity_features(csv_path, model_type=None, text_type=None):
    df = pd.read_parquet(csv_path)
    pairs = [('lda', 'title'), ('lda', 'description'), ('nmf', 'title'), ('nmf', 'description')]
    tasks = [(model_type, text_type)] if model_type and text_type else pairs

    for model_name, field in tasks:
        col_base = f'base_{field}'
        col_cand = f'cand_{field}'

        df = df[df[col_base].notna() & df[col_cand].notna()]
        all_texts = pd.concat([df[col_base], df[col_cand]]).dropna().unique()

        print(f"\n==> Обработка {model_name.upper()} для поля '{field}' — {len(all_texts)} уникальных текстов")

        preproc_map = {}
        for text in tqdm(all_texts, desc="Предобработка"):
            tokens = preprocess_ru(text)
            if tokens:
                preproc_map[text] = tokens

        if not preproc_map:
            raise ValueError("После предобработки не осталось ни одного текста")

        texts_filtered = list(preproc_map.keys())
        tokens_filtered = list(preproc_map.values())

        if model_name == 'lda':
            dictionary = corpora.Dictionary(tokens_filtered)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokens_filtered]
            topic_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=10)

            topic_vec_map = {}
            for text, tokens in tqdm(zip(texts_filtered, tokens_filtered), desc="LDA topic векторы", total=len(texts_filtered)):
                bow = dictionary.doc2bow(tokens)
                topic_dist = topic_model.get_document_topics(bow, minimum_probability=0)
                topic_vec_map[text] = np.array([prob for _, prob in topic_dist])

        elif model_name == 'nmf':
            joined_texts = [' '.join(tokens) for tokens in tokens_filtered]
            tfidf_vectorizer = TfidfVectorizer()
            tfidf = tfidf_vectorizer.fit_transform(joined_texts)

            topic_model = NMF(n_components=20, init='nndsvda', random_state=42)
            topic_distributions = topic_model.fit_transform(tfidf)

            topic_vec_map = dict(zip(texts_filtered, topic_distributions))

        else:
            raise ValueError("model_type должен быть 'lda' или 'nmf'")

        def compute_similarity(base, cand):
            vec1 = topic_vec_map.get(base)
            vec2 = topic_vec_map.get(cand)
            if vec1 is None or vec2 is None:
                return np.nan, np.nan
            return 1 - cosine(vec1, vec2), 1 - jensenshannon(vec1, vec2)

        cosine_col = f'{model_name}_{field}_cosine_sim'
        js_col = f'{model_name}_{field}_js_sim'

        tqdm.pandas(desc=f"Подсчёт {cosine_col}")
        df[[cosine_col, js_col]] = df.progress_apply(
            lambda row: compute_similarity(row[col_base], row[col_cand]),
            axis=1, result_type='expand'
        )

    return df

def run_topic_similarity_pipeline(csv_path, model_type=None, text_type=None):
    df_with_features = add_topic_similarity_features(csv_path, model_type, text_type)

    # Генерируем новое имя файла
    base, ext = os.path.splitext(csv_path)
    output_path = f"{base}_similarity_features{ext}"

    # Сохраняем результат
    df_with_features.to_parquet(output_path, index=False)
    print(f"Результат сохранён в файл: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Добавление признаков топик-сходства (LDA/NMF) к паре текстов.")
    parser.add_argument("csv_path", type=str, help="Путь к CSV-файлу с данными")
    parser.add_argument("--model_type", type=str, choices=["lda", "nmf"], default=None, help="Тип модели: lda или nmf")
    parser.add_argument("--text_type", type=str, choices=["title", "description"], default=None, help="Тип текста: title или description")

    args = parser.parse_args()
    run_topic_similarity_pipeline(args.csv_path, args.model_type, args.text_type)
