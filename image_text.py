import os
import cv2
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR

def batch_ocr(image_paths, ocr):
    """
    Обрабатываем batch изображений, возвращаем список текстов.
    Если файл не найден или битый — возвращаем пустую строку.
    """
    images_to_process = []
    indices_map = []  # Чтобы знать, к каким позициям в batch относится каждый валидный файл

    for idx, path in enumerate(image_paths):
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        images_to_process.append(path)
        indices_map.append(idx)

    texts = [""] * len(image_paths)  # Изначально пустые

    if not images_to_process:
        return texts

    try:
        ocr_results = ocr.ocr(images_to_process, cls=True)
    except Exception as e:
        print(f"Ошибка OCR: {e}")
        return texts

    for idx, result in zip(indices_map, ocr_results):
        lines = []
        for line in result:
            _, (text, _) = line
            lines.append(text)
        texts[idx] = "\n".join(lines)

    return texts

def process_parquet(df_path, images_folder, batch_size=16, save_path=None):
    df = pd.read_parquet(df_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='ru', use_gpu=True)

    base_image_paths = [os.path.join(images_folder, f"{h}.jpg") for h in df['base_image_title']]
    cand_image_paths = [os.path.join(images_folder, f"{h}.jpg") for h in df['cand_image_title']]

    base_texts = []
    cand_texts = []

    num_batches = (len(df) + batch_size - 1) // batch_size

    print("📦 Обработка изображений с OCR:")
    for i in tqdm(range(0, len(df), batch_size), total=num_batches, desc="Обработка батчей"):
        base_batch = base_image_paths[i:i+batch_size]
        cand_batch = cand_image_paths[i:i+batch_size]

        base_texts.extend(batch_ocr(base_batch, ocr))
        cand_texts.extend(batch_ocr(cand_batch, ocr))

    df['base_image_text'] = base_texts
    df['cand_image_text'] = cand_texts

    if save_path:
        df.to_parquet(save_path, index=False)
        print(f"✅ Сохранено в: {save_path}")

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR из изображений по хэшам из parquet файла")
    parser.add_argument("df_path", help="Путь к parquet файлу с данными")
    parser.add_argument("images_folder", help="Путь к папке с изображениями")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча для OCR")
    parser.add_argument("--save_path", type=str, default=None, help="Куда сохранить результат parquet")

    args = parser.parse_args()

    process_parquet(args.df_path, args.images_folder, args.batch_size, args.save_path)
