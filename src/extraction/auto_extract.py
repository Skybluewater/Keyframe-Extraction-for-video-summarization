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
from HybridImpl import Multiplication, Concatenate, Minus, Average, Attention, Division, LinearTransformation, CBP
from ExtractionKMeans import KMeans_Extraction_Impl
from ExtractionSpectral import Spectral_Clustering_Impl
from Evaluation import evaluation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

# read model config and load model, default is clip model
model_name = config.get('Settings', 'model_name', fallback='ViT-B/32')
model, device, preprocess = None, None, None

device = "cuda" if torch.cuda.is_available() else "cpu"

if model_name == "ViT-B/32":
    model, preprocess = clip.load("ViT-B/32", device=device)
elif model_name == "BAAI/BGE-VL-large":
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device)
    model.set_processor(model_name)
    model.eval()
elif model_name.split("-")[0] == "LongCLIP":
    from longclip_model import longclip
    if model_name == "LongCLIP-B":
        model, preprocess = longclip.load(r"E:\model_cache\longclip-B.pt", device=device)
    elif model_name == "LongCLIP-L":
        model, preprocess = longclip.load(r"E:\model_cache\longclip-L.pt", device=device)
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
    
    text_feature = None
    # Embed the text using the CLIP model
    with torch.no_grad():
        # ensure the text is short enough to be tokenize
        max_l = 300
        retry_cnt = 0
        while retry_cnt < 3:
            try:
                if model_name == "ViT-B/32":
                    text_input = clip.tokenize(text_input).to(device) if text_input is not None else None
                elif model_name == "BAAI/BGE-VL-large":
                    text_input = model.encode(text=text_input) if text_input is not None else None
                elif model_name == "LongCLIP-B" or model_name == "LongCLIP-L":
                    from longclip_model import longclip
                    text_input = longclip.tokenize(text_input).to(device) if text_input is not None else None
                break
            except:
                max_l -= 30
                text_input = truncate_text(text_input, max_l)
                retry_cnt += 1
        if model_name == "ViT-B/32 or model_name == LongCLIP-B" or model_name == "LongCLIP-L":
            text_feature = model.encode_text(text_input) if text_input is not None else None
        elif model_name == "BAAI/BGE-VL-large" or model_name == "BAAI/BGE-VL-base":
            text_feature = text_input

    text_feature = text_feature.cpu().numpy() if text_feature is not None else None
    return text_feature


def cal_similarity(f1, f2):
    f1 = f1.flatten()
    f2 = f2.flatten()
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def get_joint_method():
    method = args.joint.lower()
    if method == "minus":
        return Minus
    elif method == "multiplication":
        return Multiplication
    elif method == "concatenate":
        return Concatenate
    elif method == "average":
        return Average
    elif method == "attention":
        return Attention
    elif method == "division":
        return Division
    elif method == "linear":
        return LinearTransformation
    elif method == "cbp":
        return CBP
    raise ValueError(f"Invalid joint method: {method}")


def keyframe_extraction(dir_path, video_path):
    log.info(f"Processing video: {os.path.basename(video_path)}")
    basename = os.path.splitext(os.path.basename(video_path))[0]
    scenes_path = os.path.join(dir_path, f"scenes.json")
    features_path = os.path.join(dir_path, f"embeddings_{model_name.split("/")[0]}.npy")
    chunk_path = os.path.join(dir_path, "clips.json")
    save_path_txt = os.path.join(dir_path, f"res_{model_name.split("/")[0]}_{args.threshold}.txt")
    save_path_json = os.path.join(dir_path, f"res_{model_name.split("/")[0]}.json")
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        scenes = json.load(file)
        for scene in scenes['scene']:
            number_list.append(scene['start_frame'])
            number_list.append(scene['end_frame'])

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
    key_frame_per_shot = []
    redundant_delimination_shot = []
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
        hybrid_method = get_joint_method()
        start = number_list[i]
        end = number_list[i + 1]
        sub_features_img = features[start:end]
        if text_feature_e is None and text_feature_r is None:
            # best_labels, best_centers, k, index = KMeans_Extraction_Impl.clustering(sub_features_img)
            best_labels, best_centers, k, index = Spectral_Clustering_Impl.clustering(sub_features_img, image_features=sub_features_img) # 
        else:
            # Combine image features with text features
            combined_features = []
            for img_feature in sub_features_img:
                if text_feature_r is not None:
                    similarity = cal_similarity(img_feature, text_feature_r)
                    modified_text_feature_r = text_feature_r * similarity
                    combined_r = hybrid_method.hybrid_features(img_feature, modified_text_feature_r, img=args.weights[0], text=args.weights[1])
                    combined_r = combined_r / np.linalg.norm(combined_r)
                    combined_features.append(combined_r)
            combined_features = np.array(combined_features)
            best_labels, best_centers, k, index = Spectral_Clustering_Impl.clustering(combined_features, hybrid_features=combined_features, image_features=sub_features_img, text_features=text_feature_r)
        
        final_index = [x + start for x in index]
        log.info(f"Clustering result: {final_index}")
        redundant = final_index.copy()
        # The `redundant_index` variable in the code is used to store the indices of keyframes that
        # are identified as redundant during the keyframe extraction process. These redundant
        # keyframes are identified based on a specified redundancy threshold and are removed from the
        # final list of keyframes. The `redundant_index` list keeps track of these redundant keyframe
        # indices for further analysis or processing if needed.
        redundant_index.append(redundant)
        
        key_frame_per_shot.append(final_index.copy())
        final_index = list(redundancy(video_path, final_index, args.threshold, keyframe_index, text=clip_text_r))
        log.info(f"Redundant keyframe index: {final_index}")
        
        redundant_delimination_shot.append(final_index.copy())
        keyframe_index += final_index
    
    keyframe_index.sort()
    log.info(f"Final keyframe index: {str(keyframe_index)}")
    log.info(f"Redundant keyframe index: {str(redundant_index)}")
    
    # Save mid-process files
    json_file = {
        "keyframes_index": keyframe_index,
        "redundancy_delete": redundant_delimination_shot,
        "redundant_index": redundant_index
    }
    with open(save_path_txt, 'w') as f:
        for index in keyframe_index:
            f.write(f"{index}\n")

    def default_dump(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(save_path_json, "w", encoding="utf-8") as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4, default=default_dump)
    
    # Evaluation
    label_path = os.path.join(dir_path, "label.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read()
    label_idx = [int(i) for i in filter(lambda x: x != '', content.split('\n'))]
    evaluation(keyframe_index, label_idx, video_path, dir_path)


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
    parser.add_argument('--threshold', type=float, default=0.8, help="Redundancy threshold")
    parser.add_argument('--joint', type=str, default="Multiplication", help="Joint method")
    parser.add_argument('--weights', type=float, nargs='+', default=[0.5, 0.5], help="Attention weights")
    args = parser.parse_args()
    log.info("Args: " + str(args))
    log.info("weights: " + str(args.weights))
    log.info(f"Threshold: {args.threshold}")
    main(args.file_dir)