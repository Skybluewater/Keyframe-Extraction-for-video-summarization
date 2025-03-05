import os
import logging
import argparse


from utils import split_video_frames_by_duration, extract_keyframes as extract_keyframes_utils

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
def extract_frames_from_video(video_file_dir):
    # Iterate over video files in the directory
    for file in os.listdir(video_file_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_file_dir, file)
            video_name = os.path.splitext(file)[0]
            output_dir = os.path.join(video_file_dir, video_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            log.info(f"Extracting frames from {video_path} into {output_dir}")
            split_video_frames_by_duration(video_path, output_dir=output_dir)

"""
    After you have extracted the frames from the video files
    you should extract the keyframes from the frames
"""
def extract_keyframes(frame_file_dir):
    keyframes_dir = frame_file_dir
    if not os.path.exists(keyframes_dir):
        os.makedirs(keyframes_dir)
    log.info(f"Extracting keyframes from {frame_file_dir} into {keyframes_dir}")
    keyframes = extract_keyframes_utils(frame_file_dir, output_dir=keyframes_dir)
    return keyframes

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract frames from video files dir")
    parser.add_argument("--file_dir", type=str, required=True, help="Video file dir path")
    args = parser.parse_args()
    video_file_dir = args.file_dir
    extract_frames_from_video(video_file_dir)
    for dir_name in os.listdir(video_file_dir):
        dir_path = os.path.join(video_file_dir, dir_name)
        if os.path.isdir(dir_path):
            extract_keyframes(dir_path)
