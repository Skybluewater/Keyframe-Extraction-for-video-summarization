import os
import json
import ffmpeg
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# assume u have transnetv2 installed

def split_video(video_file_path, **kwargs):
    video_file_name = os.path.basename(video_file_path)
    video_file_path_dir = os.path.dirname(video_file_path)
    base_name = os.path.splitext(video_file_name)[0]

    # Split video using TransNet
    cmd = f"transnetv2_predict {video_file_path} --visualize"
    os.system(cmd)
    scenes_file_path = os.path.join(video_file_path_dir, f"{video_file_name}.scenes.txt")
    with open(scenes_file_path, 'r') as scenes_file:
        scenes_data = scenes_file.read()
    
    # Get video fps
    probe = ffmpeg.probe(video_file_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['r_frame_rate'])

    # Process scenes data
    scenes = [list(map(int, scene.split())) for scene in scenes_data.split('\n') if scene]
    scenes_timestamps = [(start / fps, end / fps) for start, end in scenes]
    scenes_dict = {f"scene": []}
    
    # Zip scenes and scenes_timestamps together
    for scene, timestamp in zip(scenes, scenes_timestamps):
        scenes_dict["scene"].append({
            "start_frame": scene[0],
            "end_frame": scene[1],
            "start_time": timestamp[0],
            "end_time": timestamp[1]
        })

    # Save scenes data to JSON
    output_file_path = os.path.join(video_file_path_dir, "scenes.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(scenes_dict, output_file, indent=4)
    log.info(f"{base_name}'s scene data saved to {output_file_path}")
    
    # Create a new folder with the name of the video file if it does not exist
    new_folder_path = os.path.join(video_file_path_dir, base_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        log.info(f"Created directory: {new_folder_path}")
    
    # Move related files to the new folder
    related_files = [
        f"{video_file_name}.scenes.txt",
        f"{video_file_name}.vis.png",
        f"{video_file_name}.predictions.txt",
        "scenes.json"
    ]
    
    import shutil
    for file_name in related_files:
        src_file_path = os.path.join(video_file_path_dir, file_name)
        dst_file_path = os.path.join(new_folder_path, file_name.replace(".mp4", ""))
        if os.path.exists(src_file_path):
            shutil.move(src_file_path, dst_file_path)
            log.info(f"Moved {src_file_path} to {dst_file_path}")
    return scenes_timestamps


if __name__ == "__main__":
    video_path = "./test/test.mp4"
    split_video(video_path)