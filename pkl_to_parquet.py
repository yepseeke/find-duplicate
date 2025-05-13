import os
import argparse
import pandas as pd

def convert_pkl_to_parquet(folder_path):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Путь '{folder_path}' не является папкой.")

    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    if not files:
        print("В папке нет .pkl файлов.")
        return

    for filename in files:
        pkl_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"[Ошибка] Не удалось загрузить {filename}: {e}")
            continue

        parquet_filename = os.path.splitext(filename)[0] + '.parquet'
        parquet_path = os.path.join(folder_path, parquet_filename)

        try:
            df.to_parquet(parquet_path, index=False)
            print(f"[✓] Сохранён: {parquet_filename}")
        except Exception as e:
            print(f"[Ошибка] Не удалось сохранить {parquet_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертация .pkl файлов в .parquet в указанной папке.")
    parser.add_argument("folder", type=str, help="Путь к папке с .pkl файлами")

    args = parser.parse_args()
    convert_pkl_to_parquet(args.folder)
