import os
import shutil
import argparse

def copy_res_files(src_dir, dst_dir, patterns):
    # Walk through the source folder
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if file starts with any of the given patterns
            if any(file.startswith(pattern) for pattern in patterns):
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
    parser = argparse.ArgumentParser(description="Copy files starting with specified patterns while preserving folder structure.")
    parser.add_argument("src", type=str, help="Source directory")
    parser.add_argument("dst", type=str, help="Destination directory")
    # Enable multiple patterns to be passed as arguments
    parser.add_argument("patterns", type=str, nargs="+", help="Patterns to match file names (e.g., res)")
    # New optional argument to show an example usage
    parser.add_argument("--example", action="store_true", help="Show an example of the argument list and exit")
    args = parser.parse_args()
    
    if args.example:
        print("Example usage:")
        print("python copy_res_files.py d:\\xue\\grd\\Keyframe-Extraction-for-video-summarization\\src\\resources d:\\xue\\grd\\Keyframe-Extraction-for-video-summarization\\dest res")
        return
    
    copy_res_files(args.src, args.dst, args.patterns)

if __name__ == '__main__':
    main()
