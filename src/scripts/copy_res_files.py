import os
import shutil
import argparse

def copy_res_files(src_dir, dst_dir, pattern="res"):
    # Walk through the source folder
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.startswith(pattern):
                src_path = os.path.join(root, file)
                # Get the relative path from the source folder
                rel_folder = os.path.relpath(root, src_dir)
                # Create the corresponding destination folder structure
                dst_folder = os.path.join(dst_dir, rel_folder)
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="Copy files starting with 'res' while preserving folder structure.")
    parser.add_argument("src", type=str, help="Source directory")
    parser.add_argument("dst", type=str, help="Destination directory")
    parser.add_argument("pattern", type=str, default="res", help="Pattern to match the files")
    args = parser.parse_args()
    copy_res_files(args.src, args.dst, args.pattern)

if __name__ == '__main__':
    main()
