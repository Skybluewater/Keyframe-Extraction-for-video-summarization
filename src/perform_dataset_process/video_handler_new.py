import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from utils import *

def main(file_dir):
    for item in os.listdir(file_dir):
        if item.endswith('.mp4'):
            basename = os.path.splitext(item)[0]
            dir_path = os.path.join(file_dir, basename)
            # first handle all things with video
            # split_video(os.path.join(file_dir, item))
            embedding_frames(os.path.join(file_dir, item))
            
            # then handle all things with audio
            # extract_audio_from_video(os.path.join(file_dir, item))
            # extract_text_from_audio(os.path.join(dir_path, "audio.wav"))
            # align_chunks_with_timestamps(os.path.join(dir_path, "text_3.json"))
            
            # finally align video clip with chunk
            # align_clip_with_chunk(os.path.join(file_dir, item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video files in a directory')
    parser.add_argument('file_dir', type=str, help='directory containing video files')
    args = parser.parse_args()
    main(args.file_dir)