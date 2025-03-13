import os
import cv2
import numpy as np
import torch
import json
import argparse
import configparser
import logging
from skimage.metrics import structural_similarity as ssim
from transformers import AutoModel
from keybert import KeyBERT

config = configparser.ConfigParser()
config.read('config.ini')

# read model config and load model, default is clip model
model_name = config.get('Settings', 'model_name', fallback='ViT-B/32')

# read BAAI model_name
device = "cuda" if torch.cuda.is_available() else "cpu"
baai_model_name = "BAAI/BGE-VL-large"
model = AutoModel.from_pretrained(baai_model_name, trust_remote_code=True, device_map=device)
model.set_processor(baai_model_name)
model.eval()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kw_model = KeyBERT(model="all-mpnet-base-v2")


def redundancy(video_path, keyframe_index, threshold, text=None):
    # colour histogram
    def color_histogram(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    init_number = 0
    final_index = []
    final_index.append(keyframe_index[init_number])

    # List for storing colour histograms
    histograms = []
    # open the video
    video = cv2.VideoCapture(video_path)

    # Iterate through the list of frame numbers
    frames = []
    for frame_index in keyframe_index:
        # Setting the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read current frame
        ret, frame = video.read()

        if ret:
            # Calculate the colour histogram
            hist = color_histogram(frame)
            histograms.append(hist)
            frames.append(frame)

    # Releasing the video
    video.release()
    histogram = np.array(histograms)
    new_histogram = []
    mid_index = []
    new_frames = []

    # Filter pure colour frames, low information frames
    for i in range(len(histogram)):
        peak_count = np.sum(histogram[i] > 0)
        # print(i, peak_count)
        if peak_count > 10:
            new_histogram.append(histogram[i])
            mid_index.append(keyframe_index[i])
            new_frames.append(frames[i])

    # Get the global similarity matrix
    simis = []
    for i in range(len(mid_index)):
        simi = []
        base = new_frames[i]
        for j in range(len(mid_index)):
            lat = new_frames[j]
            similarity = ssim(base, lat, multichannel=True, channel_axis=-1)
            simi.append(similarity)
        simis.append(simi)
        
    del_index = []
    
    for i in range(len(mid_index)):
        for j in range(i + 1, len(mid_index)):
            if mid_index[i] in del_index or mid_index[j] in del_index:
                continue
            if simis[i][j] > threshold:
                if (sum(simis[i]) > sum(simis[j])):
                    del_index.append(mid_index[i])
                else:
                    del_index.append(mid_index[j])
    
    set_mid_index = set(mid_index)
    set_del_index = set(del_index)
    set_index_after_ssim = set_mid_index - set_del_index
    
    if text is None or len(set_index_after_ssim) == 1:
        final_index = list(set_index_after_ssim)
        final_index.sort()
        return final_index
    
    mid_index = sorted(list(set_index_after_ssim))
    # extract keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
    keywords = ", ".join(list(map(lambda x: x[0], keywords)))
    query_text = "Make the image contain following objects as much as possible: " + keywords
    accumulated_scores = np.zeros(len(mid_index))
    
    # use BGE-VL-large to filter out redundant frames again
    # The first address is the address with frames extracted in it
    frame_base_address = os.path.join("./Dataset", os.path.splitext(os.path.basename(video_path))[0])
    def convert_index(index):
        # the index is of 04d format...
        return "%04d" % index
    frames_address = [os.path.join(frame_base_address, f"frame_{convert_index(index)}.png") for index in mid_index]
    with torch.no_grad():
        for index in mid_index:
            frame_address = os.path.join(frame_base_address, f"frame_{convert_index(index)}.png")
            query = model.encode(
                images = [frame_address], 
                text = [query_text]
            )

            candidates = model.encode(
                images = frames_address
            )
            
            scores = query @ candidates.T
            scores = scores.cpu().numpy().flatten()
            accumulated_scores += scores

    average_score = np.mean(accumulated_scores)
    final_index = [mid_index[i] for i in range(len(accumulated_scores)) if accumulated_scores[i] > average_score]
    final_index.sort()
    return final_index


def handle_video(dir_path, video_path):
    logger.info(f"Processing {video_path}")
    res_path = os.path.join(dir_path, f"res_{model_name.split("/")[0]}.json")
    clips_path = os.path.join(dir_path, "clips.json")
    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    with open(clips_path, "r", encoding="utf-8") as f:
        clips = json.load(f)['aligned']
    try:
        redundant_indexes = res["redundant_index"]
    except:
        redundant_indexes = res["keyframe_index"]
    assert len(redundant_indexes) == len(clips)
    redundant_res = []
    for i in range(len(redundant_indexes)):
        redundant_index = redundant_indexes[i]
        text = clips[i]["rough_chunk"]["text"]
        text = text if text != "" else None
        redundant_res.extend(redundancy(video_path, redundant_index, args.threshold, text))
    redundancy_save_path = os.path.join(dir_path, f"res_{model_name.split("/")[0]}_{args.threshold}.txt")
    with open(redundancy_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, redundant_res)))
    logger.info(f"{redundant_res}")
    return redundant_res


def main(file_dir):
    for item in os.listdir(file_dir):
        if item.endswith(".mp4"):
            basename = os.path.splitext(os.path.basename(item))[0]
            dir_path = os.path.join(file_dir, basename)
            video_path = os.path.join(file_dir, item)
            handle_video(dir_path, video_path)


def main2():
    video_path = "./Dataset2/-esJrBWj2d8.mp4"
    keyframe_index = [1945, 1957, 1987]
    threshold = 0.8
    text = "It seems that it will be impossible, but with love and patience, it can be done. Sometimes your dog or cat may be uncomfortable when you are handling parts of the body that he isn't used to having touched, get him to lie down, try to relax him and slowly get him used to allowing you to touch his ears, legs, feet and paws."
    redundancy(video_path, keyframe_index, threshold, text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir", type=str, help="Dataset dir")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for ssim")
    args = parser.parse_args()
    main(args.file_dir)