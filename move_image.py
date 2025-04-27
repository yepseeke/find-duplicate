import os
import sys
import shutil

from tqdm import tqdm


def move_files_to_new_folder(root_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    root_folder_list_dir = os.listdir(root_folder)

    for folder_to_move in root_folder_list_dir:
        if not os.path.isdir(folder_to_move):
            continue
        folder_to_move_path = os.path.join(root_folder, folder_to_move)
        files = os.listdir(folder_to_move_path)

        for file in tqdm(files, desc=f"Moving folder {folder_to_move}"):
            file_path = os.path.join(folder_to_move_path, file)
            shutil.move(file_path, os.path.join(destination_folder, file))


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <root_folder> <destination_folder>")
        sys.exit(1)

    root_folder = sys.argv[1]
    destination_folder = sys.argv[2]

    move_files_to_new_folder(root_folder, destination_folder)


if __name__ == "__main__":
    main()