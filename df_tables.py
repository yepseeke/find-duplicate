import pandas as pd
import argparse

def print_columns(filename):
    try:
        df = pd.read_parquet(filename)

        columns = df.columns.tolist()

        print("Столбцы в файле:")
        for column in columns:
            print(f"- {column}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Печатает названия столбцов Parquet файла')
    parser.add_argument('filename', type=str, help='Имя Parquet файла')

    args = parser.parse_args()

    print_columns(args.filename)
