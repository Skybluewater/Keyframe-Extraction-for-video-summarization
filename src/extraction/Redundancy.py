import cv2
import numpy as np
import torch
import os

from skimage.metrics import structural_similarity as ssim
from transformers import AutoModel
from keybert import KeyBERT

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "BAAI/BGE-VL-large"

model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device)
model.set_processor(model_name)
model.eval()

kw_model = KeyBERT(model="all-mpnet-base-v2")


def redundancy(video_path, keyframe_index, threshold, text=None):
    # colour histogram
    def color_histogram(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    init_number = 0
    final_index = []

    # print(keyframe_index)

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
    # print(mid_index)

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
                # print(mid_index[j], simis[i][j], mid_index[i])
                if (sum(simis[i]) > sum(simis[j])):
                    del_index.append(mid_index[i])
                else:
                    del_index.append(mid_index[j])
    
    
    # Remove frames in new_frames that are in del_index
    for index in sorted(del_index, reverse=True):
        del new_frames[mid_index.index(index)]
    
    set_mid_index = set(mid_index)
    set_del_index = set(del_index)
    set_index_after_ssim = set_mid_index - set_del_index
    
    if text is None:
        final_index = list(set_index_after_ssim)
        final_index.sort()
        return final_index
    
    mid_index = sorted(list(set_index_after_ssim))
    # extract keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
    keywords = ", ".join(list(map(lambda x: x[0], keywords)))
    query_text = "Make the image contain following objects as much as possible: " + keywords
    accumulated_scores = np.zeros(len(new_frames))
    frame_base_address = os.path.join("./Dataset", os.path.splitext(os.path.basename(video_path))[0])
    def convert_index(index):
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


if __name__ == "__main__":
    video_path = "./Dataset2/0tmA_C6XwfM.mp4"
    keyframe_index = [733, 788, 762]
    threshold = 0.8
    text = "It seems that it will be impossible, but with love and patience, it can be done. Sometimes your dog or cat may be uncomfortable when you are handling parts of the body that he isn't used to having touched, get him to lie down, try to relax him and slowly get him used to allowing you to touch his ears, legs, feet and paws."
    redundancy(video_path, keyframe_index, threshold, text)