import os
import shutil
import argparse

def handle_name(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if args.source in item:
            new_item_name = item.replace(args.source, args.to)
            new_item_path = os.path.join(dir_path, new_item_name)
            os.rename(item_path, new_item_path)


def main(file_dir):
    """
    The main function recursively walks through a directory and calls handle_name on each subdirectory.
    
    :param file_dir: The `file_dir` parameter in the `main` function is the directory path where you
    want to perform the operation. This function uses `os.walk` to traverse through the directory and
    its subdirectories, and for each directory found, it calls the `handle_name` function with the
    directory path as
    """
    for root, dirs, files in os.walk(file_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            handle_name(dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir", type=str, help="Dataset dir")
    parser.add_argument("--source", type=str,  help="source name")
    parser.add_argument("--to", type=str,  help="target name")
    args = parser.parse_args()
    main(args.file_dir)
