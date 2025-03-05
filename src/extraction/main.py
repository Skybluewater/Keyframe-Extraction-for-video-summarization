import os
import sys
import pickle
import cv2
import argparse
import numpy as np
import json
import torch
import clip
import logging
from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy
from feature_hybrid import element_wise_subtraction, element_wise_multiplication, element_wise_division, concatenate, cbp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def keyframe_extraction(dir_path, video_path):
    log.info(f"Processing video: {os.path.basename(video_path)}")
    basename = os.path.splitext(os.path.basename(video_path))[0]
    scenes_path = os.path.join(dir_path, f'{basename}.scenes.txt')
    features_path = os.path.join(dir_path, 'embeddings.npy')
    chunk_path = os.path.join(dir_path, 'clips.json')
    save_path = os.path.join(dir_path, "res.json")
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # print(line)
            numbers = line.strip().split(' ')
            # print(numbers)
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
    keyframe_index = []
    for i in range(0, len(number_list) - 1, 2):
        log.info(f"Processing shot {i // 2} with frames from {number_list[i]} to {number_list[i + 1]}")
        # get the current text description of the clip
        clip_idx = i // 2
        clip_text_e = clips[clip_idx]['exact_chunk']['text'].strip() if len(clips[clip_idx]['exact_chunk']['text']) > 0 else None
        clip_text_r = clips[clip_idx]['rough_chunk']['text'].strip() if len(clips[clip_idx]['rough_chunk']['text']) > 0 else None
        
        # Embedding the two texts separately using the CLIP model
        def ensure_models_have_less_than_75_desc(text):
            if text is None or len(text) < 250:
                return text.strip() if type(text) == str else text
            texts = text.split('.')[:-1]
            text_cnt = 0
            new_text = ""
            for text in texts:
                if len(new_text) + len(text) >= 250:
                    log.info(f"Truncated text: {new_text}")
                    return new_text + text[:300 - len(new_text)]
                new_text += text.strip() + '. '
            return new_text
        
        clip_text_e = ensure_models_have_less_than_75_desc(clip_text_e)
        clip_text_r = ensure_models_have_less_than_75_desc(clip_text_r)
        text_input_e = clip.tokenize(clip_text_e).to(device) if clip_text_e is not None else None
        text_input_r = clip.tokenize(clip_text_r).to(device) if clip_text_r is not None else None
        with torch.no_grad():
            text_feature_e = model.encode_text(text_input_e) if text_input_e is not None else None
            text_feature_r = model.encode_text(text_input_r) if text_input_r is not None else None
        text_feature_e = text_feature_e.cpu().numpy() if text_feature_e is not None else None
        text_feature_r = text_feature_r.cpu().numpy() if text_feature_r is not None else None
        
        # get current features
        start = number_list[i]
        end = number_list[i + 1]
        sub_features_img = features[start:end]
        if text_feature_e is None and text_feature_r is None:
            # continue
            best_labels, best_centers, k, index = kmeans_silhouette(sub_features_img)
        else:
            # Combine image features with text features
            combined_features = []
            for img_feature in sub_features_img:
                # if text_feature_e is not None:
                #     combined_e = element_wise_multiplication(img_feature, text_feature_e)
                #     combined_features.append(combined_e)
                if text_feature_r is not None:
                    combined_r = element_wise_multiplication(img_feature, text_feature_r)
                    combined_features.append(combined_r)
            combined_features = np.array(combined_features)
            best_labels, best_centers, k, index = kmeans_silhouette(combined_features)
            
        # print(index)
        log.info(f"Clustering result: {index}")
        log.info(f"Clustering centers: {best_centers}")
        log.info(f"Clustering labels: {best_labels}")
        final_index = [x + start for x in index]
        redundant = final_index.copy()
        # final_index.sort()
        # print("clustering：" + str(keyframe_index))
        # print(start, end)
        final_index = redundancy(video_path, final_index, 0.94)
        # print(final_index)
        keyframe_index += final_index
    keyframe_index.sort()
    log.info(f"Final keyframe index: {str(keyframe_index)}")
    res = {
        "clustering_result": index,
        "clustering_centers": best_centers,
        "clustering_labels": best_labels,
        "redundant_keyframes_idx": redundant,
        "keyframes_idx": keyframe_index
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


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