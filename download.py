import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import argparse

def download_file(url, folder):
    try:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(folder, filename)

        if os.path.exists(filepath):
            print(f"✓ Файл уже существует: {filename}")
            return True, filename

        print(f"↓ Начинаю загрузку: {filename}")

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"✓ Успешно загружено: {filename}")
        return True, filename

    except Exception as e:
        print(f"✗ Ошибка при загрузке {filename}: {str(e)}")
        return False, filename

def main():
    parser = argparse.ArgumentParser(description='Параллельная загрузка файлов из JSON')
    parser.add_argument('json_file', help='Путь к JSON файлу с URL')
    parser.add_argument('-o', '--output', default='.', help='Папка для загрузки')
    parser.add_argument('-t', '--threads', type=int, default=8, help='Количество потоков')
    args = parser.parse_args()

    try:
        with open(args.json_file) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON должен содержать словарь с категориями файлов")

        tasks = []
        for category, urls in data.items():
            if not isinstance(urls, list):
                continue
            category_dir = os.path.join(args.output, category)
            tasks.extend([(url, category_dir) for url in urls])

        print(f"Найдено {len(tasks)} файлов для загрузки в '{args.output}'")
        print(f"Используется {args.threads} параллельных потоков...\n")

        success = 0
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(download_file, *task) for task in tasks]

            for future in as_completed(futures):
                result, filename = future.result()
                if result:
                    success += 1

        print(f"\nЗавершено! Успешно: {success}/{len(tasks)}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()


