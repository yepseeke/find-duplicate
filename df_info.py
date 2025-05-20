import pandas as pd
import os
import datetime
import argparse

def print_parquet_info(file_path):
    if not file_path.endswith('.parquet'):
        raise ValueError("Ожидается файл с расширением .parquet")


    df = pd.read_parquet(file_path)

    try:
        timestamp = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(timestamp)
    except Exception as e:
        creation_date = "Не удалось получить дату: " + str(e)

    pd.set_option('display.max_columns', None)

    print(f"\nДата создания файла: {creation_date}")
    print("\nСтолбцы в DataFrame:")
    for col in df.columns:
        print(col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Показать информацию о parquet-файле")
    parser.add_argument("file_path", help="Путь к .parquet файлу")
    args = parser.parse_args()

    print_parquet_info(args.file_path)
