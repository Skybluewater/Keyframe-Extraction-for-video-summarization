import os
import logging

import utils
import argparse

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
def align_img(video_file_dir):
    # Iterate over video files in the directory
    utils.align_img_with_chunk(video_file_dir)
    log.info(f"Aligning images in {video_file_dir} is done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align images in video files directory.")
    parser.add_argument("--file_dir", type=str, required=True, help="The directory of the video files.")
    args = parser.parse_args()
    for dir_name in os.listdir(args.file_dir):
        subdir = os.path.join(args.file_dir, dir_name)
        if os.path.isdir(subdir):
            log.info(f"Aligning images in {dir_name}")
            utils.align_img_with_chunk(subdir)
