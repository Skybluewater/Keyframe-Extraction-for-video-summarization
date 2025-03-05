import cv2
import torch
import clip
import os
import re
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
import warnings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
model, preprocess = None, None

warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use the updated video processing utilities.",
    DeprecationWarning,
    stacklevel=2
)
def split_video_frames_by_duration(video_path, duration=0.3, output_dir=None):
    """
    Extracts frames from a video at specific intervals.

    Args:
        video_path (str): Path to the video file.
        duration (float): Interval in seconds to extract the frames.
        output_dir (str): Directory to save the extracted frames. If None, frames are returned.

    Returns:
        list: List of extracted frames if output_dir is None.
    """
    # Create the output folder if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * duration)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            if output_dir is None:
                frames.append(frame)
            else:
                timestamp = frame_count / fps
                frame_filename = os.path.join(output_dir, f"frame_{timestamp:.2f}.jpg")
                cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    if output_dir is None:
        return frames

def _cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

def _select_keyframes(video_dataset, model, device, preprocess, similarities_threshold=0.85):
    keyframes = []
    video_features = []
    
    with torch.no_grad():
        first_frame_data = video_dataset[0]
        first_frame = first_frame_data['img'].unsqueeze(0).to(device)
        if first_frame.shape[1] == 1:
            first_frame = first_frame.repeat(1, 3, 1, 1)
        first_frame_preprocessed = preprocess(transforms.ToPILImage()(first_frame.squeeze(0).cpu())).unsqueeze(0).to(device)
        first_frame_features = model.encode_image(first_frame_preprocessed)
        
        keyframes.append(0)
        video_features.append(first_frame_features)
        
        for idx in tqdm(range(1, len(video_dataset))):
            current_frame_data = video_dataset[idx]
            current_frame = current_frame_data['img'].unsqueeze(0).to(device)
            if current_frame.shape[1] == 1:
                current_frame = current_frame.repeat(1, 3, 1, 1)
            current_frame_preprocessed = preprocess(transforms.ToPILImage()(current_frame.squeeze(0).cpu())).unsqueeze(0).to(device)
            current_frame_features = model.encode_image(current_frame_preprocessed)
            similarity = _cosine_similarity(current_frame_features, video_features[-1])
            if similarity.item() < similarities_threshold:
                keyframes.append(idx)
                video_features.append(current_frame_features)
    
    log.info(f"Selected {len(keyframes)} keyframes.")
    return keyframes

def _load_frame_dataset(frames_folder):
    frame_dataset = []
    for file in os.listdir(frames_folder):
        if file.endswith('.jpg'):
            img = Image.open(f'{frames_folder}/{file}')
            img_tensor = transforms.ToTensor()(img)
            frame_dataset.append({'img': img_tensor, 'file_name': file})
    return frame_dataset

def _load_device_and_model(model_name="ViT-B/32"):
    global model, preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None or preprocess is None:
        model, preprocess = clip.load(model_name, device=device)
    return device, model, preprocess


warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use the updated video processing utilities.",
    DeprecationWarning,
    stacklevel=2
)
def extract_keyframes(frames_folder, **kwargs):
    device, model, preprocess = _load_device_and_model(kwargs.get("model_name", "ViT-B/32"))
    output_dir = kwargs.get('output_dir', None)
    video_dataset = _load_frame_dataset(frames_folder)
    
    similarities_threshold = kwargs.get('similarities_threshold', 0.85)
    kwargs.pop('output_dir', None)
    
    keyframes = _select_keyframes(video_dataset, model, device, preprocess, similarities_threshold, **kwargs)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx in keyframes:
            frame_data = video_dataset[idx]
            img = transforms.ToPILImage()(frame_data['img'])
            match = re.search(r'frame_(\d+\.\d+)', frame_data['file_name'])
            # Sanitize filename by stripping any embedded directory paths
            filename = os.path.basename(frame_data['file_name'])
            if match:
                filename = 'keyframe_' + match.group(1) + '.jpg'
            img.save(os.path.join(output_dir, filename))
    
    return keyframes


if __name__ == '__main__':
    video_path = "./test/video_1.mp4"
    dir = "./output_frames"
    # split_video_frames_by_duration(video_path, output_dir=output_dir)
    extract_keyframes(dir, output_dir=dir)
    log.info("Keyframes extraction completed.")