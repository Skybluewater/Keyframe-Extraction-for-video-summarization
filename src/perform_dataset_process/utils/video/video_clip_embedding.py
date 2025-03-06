import cv2
import torch
import clip
import os
import re
import configparser
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

# load model via config settings
model_name = config.get('Settings', 'model_name', fallback='ViT-B/32')
model, preprocess = None, None
log.info(f"Using model: {model_name}")

def _load_device_and_model():
    global model, preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name=="ViT-B/32":
        model, preprocess = clip.load(model_name, device=device)
        return device, model, preprocess
    else:
        from transformers import AutoModel
        # "BAAI/BGE-VL-large" or "BAAI/BGE-VL-base"
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True) # You must set trust_remote_code=True
        model.set_processor(model_name)
        model.eval()
    return device, model, None

def _extract_frames(video_file_path):
    video_file_name = os.path.splitext(os.path.basename(video_file_path))[0]
    output_dir = os.path.join(os.path.dirname(video_file_path), video_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    
    # Use ffmpeg to extract frames
    command = f"ffmpeg -i {video_file_path} {output_dir}/frame_%04d.png"
    os.system(command)
    
    # Load frames into a list
    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if re.match(r'frame_\d{4}\.png', f)])
    frames = [Image.open(frame_file) for frame_file in frame_files]
    
    return frames

def embedding_frames(video_file_path, **kwargs):
    video_file_name = os.path.basename(video_file_path)
    video_dir = os.path.dirname(video_file_path)
    
    device, model, preprocess = _load_device_and_model()

    frames = _extract_frames(video_file_path)
    
    basename = os.path.splitext(video_file_name)[0]
    output_dir = os.path.join(video_dir, basename)
    embeddings = []
    for frame in tqdm(frames, desc="Embedding frames"):
        with torch.no_grad():
            if model_name == "ViT-B/32":
                image = preprocess(frame).unsqueeze(0).to(device)
                embedding = model.encode_image(image)
            else:
                embedding = model.encode(images=frame)
        embeddings.append(embedding.cpu().numpy())
    
    embedding_file_path = os.path.join(output_dir, f"embeddings_{model_name.split("/")[0]}.npy")
    np.save(embedding_file_path, np.array(embeddings))
    return np.array(embeddings)

if __name__ == '__main__':
    video_path = "./test/test.mp4"
    embeddings = embedding_frames(video_path)
    log.info("Frame embeddings completed.")