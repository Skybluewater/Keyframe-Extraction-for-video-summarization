import os
import sys
import argparse
import numpy as np
import json
import torch
import clip
import logging
import configparser
from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy
from HybridImpl import Multiplication, Concatenate, Subtraction, Average, Attention, Division, LinearTransformation, CBP
from ExtractionKMeans import KMeans_Extraction_Impl
from ExtractionSpectral import Spectral_Clustering_Impl

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

# read model config and load model, default is clip model
model_name = config.get('Settings', 'model_name', fallback='ViT-B/32')
model, device, preprocess = None, None, None
if model_name == "ViT-B/32":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
else:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device)
    model.set_processor(model_name)
    model.eval()
log.info(f"Using model: {model_name}")


def text_embeddings(text_input, model, device):
    # Embedding the two texts separately using the CLIP model
    def truncate_text(text, max_l=300):
        if text is None or len(text) < max_l:
            return text.strip() if type(text) == str else text
        texts = text.split('.')[:-1]
        new_text = ""
        for text in texts:
            if len(new_text) + len(text) >= max_l:
                log.info(f"Truncated text: {new_text}")
                return new_text + text[:max_l - len(new_text)]
            new_text += text.strip() + '. '
        return new_text
    
    # Embed the text using the CLIP model
    with torch.no_grad():
        # ensure the text is short enough to be tokenize
        max_l = 300
        retry_cnt = 0
        while retry_cnt < 3:
            try:
                if model_name == "ViT-B/32":
                    text_input = clip.tokenize(text_input).to(device) if text_input is not None else None
                else:
                    text_input = model.encode(text=text_input) if text_input is not None else None
                break
            except:
                max_l -= 30
                text_input = truncate_text(text_input, max_l)
                retry_cnt += 1
        if model_name == "ViT-B/32":
            text_feature = model.encode_text(text_input) if text_input is not None else None
        else:
            text_feature = text_input

    text_feature = text_feature.cpu().numpy() if text_feature is not None else None
    return text_feature


def keyframe_extraction(dir_path, video_path):
    log.info(f"Processing video: {os.path.basename(video_path)}")
    basename = os.path.splitext(os.path.basename(video_path))[0]
    scenes_path = os.path.join(dir_path, f"{basename}.scenes.txt")
    features_path = os.path.join(dir_path, f"embeddings_{model_name.split("/")[0]}.npy")
    chunk_path = os.path.join(dir_path, "clips.json")
    save_path = os.path.join(dir_path, f"res_{model_name.split("/")[0]}.txt")
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            numbers = line.strip().split(' ')
            number_list.extend([int(number) for number in numbers])

    # Read inference data from local
    with open(features_path, 'rb') as f:
        features = np.load(f)
    features = np.asarray(features)
    # print(len(features))
    
    with open(chunk_path, 'r') as f:
        clips = json.load(f)
    clips = clips['aligned']
    
    # Clustering at each shot to obtain keyframe sequence numbers
    keyframe_index, redundant_index = [], []
    # process keyframe extraction for each shot
    for i in range(0, len(number_list) - 1, 2):
        log.info(f"Processing shot {i // 2} with frames from {number_list[i]} to {number_list[i + 1]}")
        # get the current text description of the clip
        clip_idx = i // 2
        clip_text_e = clips[clip_idx]['exact_chunk']['text'].strip() if len(clips[clip_idx]['exact_chunk']['text']) > 0 else None
        clip_text_r = clips[clip_idx]['rough_chunk']['text'].strip() if len(clips[clip_idx]['rough_chunk']['text']) > 0 else None
        
        # get the text embeddings
        text_feature_e = text_embeddings(clip_text_e, model, device)
        text_feature_r = text_embeddings(clip_text_r, model, device)
        
        # joint embedding current features and cluster
        hybrid_method = Multiplication
        start = number_list[i]
        end = number_list[i + 1]
        sub_features_img = features[start:end]
        if text_feature_e is None and text_feature_r is None:
            best_labels, best_centers, k, index = KMeans_Extraction_Impl.clustering(sub_features_img)
            # best_labels, best_centers, k, index = Spectral_Clustering_Impl.clustering(sub_features_img, image_features=sub_features_img) # 
        else:
            # Combine image features with text features
            combined_features = []
            for img_feature in sub_features_img:
                if text_feature_r is not None:
                    combined_r = hybrid_method.hybrid_features(img_feature, text_feature_r)
                    combined_features.append(combined_r)
            combined_features = np.array(combined_features)
            best_labels, best_centers, k, index = Spectral_Clustering_Impl.clustering(combined_features, hybrid_features=combined_features, image_features=sub_features_img, text_features=text_feature_r)
        
        final_index = [x + start for x in index]
        log.info(f"Clustering result: {final_index}")
        redundant = final_index.copy()
        redundant_index.append(redundant)
        final_index = redundancy(video_path, final_index, 0.83)
        log.info(f"Redundant keyframe index: {final_index}")
        keyframe_index += final_index
    
    keyframe_index.sort()
    log.info(f"Final keyframe index: {str(keyframe_index)}")
    log.info(f"Redundant keyframe index: {str(redundant_index)}")
    with open(save_path, 'w') as f:
        for index in keyframe_index:
            f.write(f"{index}\n")


def main(file_dir):
    for item in os.listdir(file_dir):
        if item.endswith('.mp4'):
            basename = os.path.splitext(item)[0]
            dir_path = os.path.join(file_dir, basename)
            video_path = os.path.join(file_dir, item)
            keyframe_extraction(dir_path, video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_dir', type=str, help="Dataset directory")
    args = parser.parse_args()
    main(args.file_dir)