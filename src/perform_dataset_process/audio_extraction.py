import os
import logging
import argparse


from utils import extract_audio_from_video, extract_text_from_audio, align_chunks_with_timestamps

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

"""
    This file is used to extract video frames from the video files.
    input:
        video_file_dir: the directory of the video file
    output:
        each video file should has it's own directory, and the frames are stored in the directory
        the directory name is the same as the video file name
    Requirements:
        you should use methods from utils to extract frames from the video file
        like split_video_frames_by_duration, extract_keyframes
"""
def extract_audio(video_file_dir):
    # Iterate over video files in the directory
    for file in os.listdir(video_file_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_file_dir, file)
            video_name = os.path.splitext(file)[0]
            output_dir = os.path.join(video_file_dir, video_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            log.info(f"Extracting frames from {video_path} into {output_dir}")
            extract_audio_from_video(video_path, audio_path=output_dir)


def extract_text(audio_dir):
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    audio_name = os.path.basename(audio_dir)
    audio_file_path = os.path.join(audio_dir, f"{audio_name}.wav")
    extract_text_from_audio(audio_file_path)


def extract_chunks(text_dir):
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    text_name = os.path.basename(text_dir)
    text_file_path = os.path.join(text_dir, "text_3.json")
    align_chunks_with_timestamps(text_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from video files directory.")
    parser.add_argument("--file_dir", type=str, required=True, help="The directory of the video files.")
    args = parser.parse_args()
    video_file_dir = args.file_dir
    extract_audio(video_file_dir)
    for dir_name in os.listdir(video_file_dir):
        dir_path = os.path.join(video_file_dir, dir_name)
        if os.path.isdir(dir_path):
            extract_text(dir_path)
            extract_chunks(dir_path)
