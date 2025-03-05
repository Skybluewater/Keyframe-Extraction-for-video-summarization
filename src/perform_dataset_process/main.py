import os
import argparse
import logging
from video_extraction import extract_frames_from_video, extract_keyframes
from audio_extraction import extract_audio, extract_text, extract_chunks
from align import align_img
from to_jsonl import read_vlm_data, save_data_to_jsonl

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def process_video_directory(video_file_dir):
    # Rename video files to symbol values to avoid the problem of Chinese characters
    file_mapping = dict()
    for idx, file in enumerate(os.listdir(video_file_dir)):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            file_mapping[idx] = os.path.splitext(file)[0]
            suffix = os.path.splitext(file)[1]
            new_file_name = f"{idx}{suffix}"
            os.rename(os.path.join(video_file_dir, file), os.path.join(video_file_dir, new_file_name))
    log.info(f"File mapping: {file_mapping}")
    
    # Extract frames and keyframes for each video file in the directory
    extract_frames_from_video(video_file_dir)
    for item in os.listdir(video_file_dir):
        subdir = os.path.join(video_file_dir, item)
        if os.path.isdir(subdir):
            log.info(f"Processing keyframes in {subdir}")
            extract_keyframes(subdir)
    
    # Restore video name and file name
    for file in os.listdir(video_file_dir):
        prefix = file.split(".")[0] if os.path.isfile(os.path.join(video_file_dir, file)) else file
        if prefix.isdigit() and int(prefix) not in file_mapping:
            continue
        prefix = int(prefix)
        restored_prefix = file_mapping[prefix]
        restored_name = file.replace(str(prefix), restored_prefix)
        log.info(f"Restoring name {file} to {restored_name}")
        os.rename(os.path.join(video_file_dir, file), os.path.join(video_file_dir, restored_name))
    
    # Extract audio, then process text and chunks for each video directory
    extract_audio(video_file_dir)
    for item in os.listdir(video_file_dir):
        subdir = os.path.join(video_file_dir, item)
        if os.path.isdir(subdir):
            log.info(f"Extracting text and chunks in {subdir}")
            extract_text(subdir)
            extract_chunks(subdir)
    
    # Align images with chunk info
    log.info("Aligning images with chunk info.")
    for item in os.listdir(video_file_dir):
        subdir = os.path.join(video_file_dir, item)
        if os.path.isdir(subdir):
            log.info(f"Aligning text and chunks in {subdir}")
            align_img(subdir)
    
    # Convert processed data to JSONL format (assuming to_jsonl.convert_data exists)
    log.info("Converting data to JSONL format.")
    jsonl_content: list[dict] = []
    for item in os.listdir(video_file_dir):
        subdir = os.path.join(video_file_dir, item)
        if os.path.isdir(subdir):
            log.info(f"Converting data in {subdir} to jsonl")
            jsonl_content.extend(read_vlm_data(subdir))
    save_data_to_jsonl(jsonl_content, video_file_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process videos to extract frames, keyframes, audio, and align chunks.")
    parser.add_argument("--file_dir", type=str, required=True, help="The directory containing video files.")
    args = parser.parse_args()
    process_video_directory(args.file_dir)
