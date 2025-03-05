import moviepy.editor as mp
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_audio_from_video(video_path, audio_path=None):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to save the extracted audio file.
    """
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    audio_name = "audio.wav"
    if not os.path.exists(os.path.join(os.path.dirname(video_path), base_name)):
        os.makedirs(os.path.join(os.path.dirname(video_path), base_name))
    audio_path = audio_path or os.path.join(os.path.dirname(video_path), base_name)
    audio_path = os.path.join(audio_path, audio_name)
    audio.write_audiofile(audio_path)
    return audio


if __name__ == "__main__":
    video_path = "./test/test.mp4"
    extract_audio_from_video(video_path)
    log.info("Audio extraction completed.")
