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


def redundancy(video_path, candidate_idxs, threshold, prev_idxs, text=None):
    # colour histogram
    def color_histogram(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    # List for storing colour histograms
    histograms = []
    # open the video
    video = cv2.VideoCapture(video_path)

    # Iterate through the list of frame numbers
    candidate_frames = []
    for candidate_idx in candidate_idxs:
        # Setting the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)

        # Read current frame
        ret, frame = video.read()

        if ret:
            # Calculate the colour histogram
            hist = color_histogram(frame)
            histograms.append(hist)
            candidate_frames.append(frame)
    
    prev_frames = []
    for prev_idx in prev_idxs:
        # Setting the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)

        # Read current frame
        ret, frame = video.read()

        if ret:
            # Calculate the colour histogram
            prev_frames.append(frame)

    # Releasing the video
    video.release()
    histogram = np.array(histograms)

    def filter_low_info_frames(candidate_idxs, candidate_frames, histograms, info_threshold=10):
        filtered_candidate_idxs = []
        filtered_candidate_frames = []
        for i in range(len(histograms)):
            peak_count = np.sum(histograms[i] > 0)
            if peak_count > info_threshold:
                filtered_candidate_idxs.append(candidate_idxs[i])
                filtered_candidate_frames.append(candidate_frames[i])
        return filtered_candidate_idxs, filtered_candidate_frames

    _candidate_idxs, _candidate_frames = filter_low_info_frames(candidate_idxs, candidate_frames, histogram)
    if len(_candidate_frames) == 0:
        return []

    # Nested function to filter redundant frames with previous frames
    def filter_with_prev_frames(candidate_idxs, candidate_frames, prev_frames, threshold):
        filtered_candidate_idxs = []
        filtered_candidate_frames = []
        for idx, frame in zip(candidate_idxs, candidate_frames):
            # Check if the candidate has sufficient similarity with any previous frame
            similarities = [ssim(frame, prev, multichannel=True, channel_axis=-1) for prev in prev_frames]
            if max(similarities) < 0.85:
                filtered_candidate_idxs.append(idx)
                filtered_candidate_frames.append(frame)
        return filtered_candidate_idxs, filtered_candidate_frames

    if len(prev_frames) > 0:
        _candidate_idxs, _candidate_frames = filter_with_prev_frames(_candidate_idxs, _candidate_frames, prev_frames, threshold)
    if len(_candidate_frames) == 0:
        return []
    
    def iterative_redundant_deletion(candidate_idxs, candidate_frames, threshold):
        # Get the global similarity matrix
        simis = []
        for i in range(len(candidate_idxs)):
            simi = []
            base = candidate_frames[i]
            for j in range(len(candidate_idxs)):
                lat = candidate_frames[j]
                similarity = ssim(base, lat, multichannel=True, channel_axis=-1)
                simi.append(similarity)
            simis.append(simi)

        # Iterative redundant deletion algorithm
        keep = list(range(len(candidate_idxs)))
        changed = True
        while changed:
            changed = False
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    if simis[keep[i]][keep[j]] > threshold:
                        # Remove the frame with higher total similarity
                        if sum(simis[keep[i]]) > sum(simis[keep[j]]):
                            remove_idx = keep[i]
                            for k in range(len(simis[remove_idx])):
                                simis[remove_idx][k] = 0
                            for k in range(len(simis)):
                                simis[k][remove_idx] = 0
                        else:
                            remove_idx = keep[j]
                            for k in range(len(simis[remove_idx])):
                                simis[remove_idx][k] = 0
                            for k in range(len(simis)):
                                simis[k][remove_idx] = 0
                        keep.remove(remove_idx)
                        changed = True
                        break
                if changed:
                    break
        return {candidate_idxs[idx] for idx in keep}

    set_index_after_ssim = iterative_redundant_deletion(_candidate_idxs, _candidate_frames, threshold)
    
    if len(set_index_after_ssim) <= 1:
        return list(set_index_after_ssim)
    
    def convert_index(index):
        # the index is of 04d format...
        return "%04d" % index
    # redundant delimination with text
    _candidate_idxs = sorted(list(set_index_after_ssim))
    accumulated_scores = np.zeros(len(_candidate_idxs))
    frame_base_address = os.path.join("./Dataset", os.path.splitext(os.path.basename(video_path))[0])
    frames_address = [os.path.join(frame_base_address, f"frame_{convert_index(index)}.png") for index in _candidate_idxs]
    
    if text is None:
        def filter_redundant_frames_baai_image_only():
            with torch.no_grad():
                for idx, _candidate_idx in enumerate(_candidate_idxs):
                    # even though frame_address is not used in encoding, it's defined here for consistency
                    frame_address = os.path.join(frame_base_address, f"frame_{convert_index(_candidate_idx)}.png")
                    query = model.encode(
                        text = [query_text],
                        images = [frame_address]
                    )
                    candidates = model.encode(
                        images = frames_address
                    )
                    scores = query @ candidates.T
                    scores = scores.cpu().numpy().flatten()
                    scores[idx] = 0
                    accumulated_scores += scores
                return accumulated_scores
        
        accumulated_scores = filter_redundant_frames_baai_image_only()
    else:    
        # extract keywords
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')[:3]
        query_text = "Photo with such elements: "
        
        def filter_redundant_frames_baai(query_text, accumulated_scores):
            # use BGE-VL-large to filter out redundant frames again
            with torch.no_grad():
                for idx, keyword in enumerate(keywords):
                    query = model.encode(
                        text = query_text + keyword[0]
                    )
                    candidates = model.encode(
                        images = frames_address
                    )
                    scores = query @ candidates.T
                    scores = scores.cpu().numpy().flatten()
                    accumulated_scores += scores * keyword[1]
            return accumulated_scores

        accumulated_scores = filter_redundant_frames_baai(query_text, accumulated_scores)

    if len(accumulated_scores) == 2:
        if (np.max(accumulated_scores) - np.min(accumulated_scores)) / np.min(accumulated_scores) < 0.5 and np.max(accumulated_scores) >= 0.2:
            return [_candidate_idxs[np.argmax(accumulated_scores)]]
        else:
            return _candidate_idxs
    
    # Normalize accumulated_scores using min-max normalization
    min_val = np.min(accumulated_scores)
    max_val = np.max(accumulated_scores)
    if max_val != min_val:
        accumulated_scores = (accumulated_scores - min_val) / (max_val - min_val)
    else:
        accumulated_scores = np.zeros_like(accumulated_scores)
    
    order = np.argsort(accumulated_scores)
    accumulated_scores = accumulated_scores[order]
    _candidate_idxs = [ _candidate_idxs[i] for i in order ]
    
    def longest_valid_subsequence(seq, threshold):
        n = len(seq)
        if n == 0:
            return []

        # dp[i] is the length of the longest subsequence ending at index i.
        dp = [1] * n
        # prev[i] helps us trace back the subsequence.
        prev = [-1] * n

        # Loop through every pair of indices to update the longest subsequence.
        for i in range(n):
            for j in range(i):
                # Check if the gap between seq[i] and seq[j] is more than threshold.
                if abs(seq[i] - seq[j]) > threshold:
                    # If appending seq[i] to the subsequence ending at j increases the length, update.
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

        # Find the index where the longest subsequence ends.
        max_index = max(range(n), key=lambda i: dp[i])
        
        # Reconstruct the subsequence by tracing back through 'prev'.
        subseq = []
        i = max_index
        while i != -1:
            subseq.append(i)
            i = prev[i]
        subseq.reverse()
        return subseq
    
    valid_ss = longest_valid_subsequence(accumulated_scores, np.log2(len(accumulated_scores)) / len(accumulated_scores))
    
    final_index = [_candidate_idxs[i] for i in valid_ss]
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