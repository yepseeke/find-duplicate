import zipfile
import os
import sys
from concurrent.futures import ThreadPoolExecutor


def unzip_file(zip_path, base_path):
    folder_name = os.path.join(base_path, os.path.basename(zip_path).rsplit('.', 1)[0])
    if os.path.exists(folder_name):
        print(f"Пропущено (папка уже существует): {folder_name}")
        return
    os.makedirs(folder_name, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(folder_name)
    print(f"Распакован: {zip_path} → {folder_name}")


def main():
    if len(sys.argv) < 2:
        print("❗ Укажите путь к папке с архивами как аргумент!")
        print("Пример: python3 unzip_all.py /путь/к/папке")
        sys.exit(1)

    base_path = sys.argv[1]

    if not os.path.isdir(base_path):
        print(f"❗ Ошибка: '{base_path}' — это не папка.")
        sys.exit(1)

    zip_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.zip')]

    if not zip_files:
        print("❗ В указанной папке нет zip-файлов.")
        sys.exit(0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for zip_path in zip_files:
            executor.submit(unzip_file, zip_path, base_path)


if __name__ == "__main__":
    main()
