import os
import copy
import json
import cv2
import argparse
import numpy as np


def evaluation(keyframes_idx, test_index, video_path, dir_path):
    save_path = os.path.join(dir_path, "test_result.json")
    def color_histogram(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        return hist.flatten()

    # fidelity and ratio
    def fidelity_and_ratio(features, true_keyframe, keyframe_index):
        # print(len(keyframe_index))
        # print(len(features))
        #  calculate ratio
        ratio = 1 - (len(keyframe_index) / len(features))
        # print(ratio)

        true_features = []
        for i in range(len(true_keyframe)):
            true_features.append(features[true_keyframe[i]])

        keyframe_features = []
        for j in range(len(keyframe_index)):
            keyframe_features.append(features[keyframe_index[j]])

        # calculate fidelity
        dist = []
        dist_max = []
        for m in range(len(keyframe_features)):
            d_min = float('inf')
            d_max = 0
            for n in range(len(true_features)):
                fir = keyframe_features[m]
                sec = true_features[n]
                # normalisation
                cv2.normalize(fir, fir, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(fir)
                cv2.normalize(sec, sec, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(sec)
                # Calculating cosine similarity
                if np.all(fir == 0) or np.all(sec == 0):
                    similarity = 0
                else:
                    similarity = np.dot(fir, sec) / (np.linalg.norm(fir) * np.linalg.norm(sec))

                # print(similarity)
                if similarity < d_min:
                    d_min = similarity

            dist.append(d_min)

        dist.sort(reverse=True)
        d_max = dist[0]
        fidelity = 1 - d_max
        return fidelity, ratio

    # Matching stage:
    # read video
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frames.append(frame)

    features = []
    for frame in frames:
        # color histogram
        hist = color_histogram(frame)
        features.append(hist)

    x_num = 0
    match_index = []
    lens_key = len(keyframes_idx)
    lens_text = len(test_index)
    keyframe_center_copy = copy.deepcopy(keyframes_idx)
    text_index_copy = copy.deepcopy(test_index)

    # Get the similarity matrix
    simis = []
    for i in range(lens_key):
        simi = []
        base = features[keyframe_center_copy[i]]
        cv2.normalize(base, base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        for j in range(lens_text):
            lat = features[text_index_copy[j]]
            cv2.normalize(lat, lat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            similarity = np.dot(base, lat) / (np.linalg.norm(base) * np.linalg.norm(lat))
            simi.append(similarity)
        simis.append(simi)
    # matching
    matchs = []
    while lens_key > 0 and lens_text > 0:
        max_num = float('-inf')  # Initialise the maximum number to negative infinity
        max_index = None
        # Iterate through the entire array to find the maximum value
        for num_i, row in enumerate(simis):
            for num_j, num in enumerate(row):
                if num > max_num:
                    max_num = num
                    max_index = (num_i, num_j)
        i, j = max_index
        if max_num > 0.9:
            new_i = keyframes_idx[i]
            new_j = test_index[j]
            match = (new_i, new_j)
            match_index.append(new_j)
            matchs.append(match)

        for row in simis:
            row[j] = -1
        simis[i] = [-1] * len(simis[i])
        lens_key -= 1
        lens_text -= 1
    matchs = sorted(matchs)
    match_index = sorted(match_index)
    x_num = len(matchs)
    print("match_index:" + str(match_index))
    print("match:" + str(len(matchs)) + ":" + str(matchs))

    # 计算f值
    print(len(test_index))
    print(len(features))
    procession = float(x_num / len(test_index))
    recall = float(x_num / len(keyframes_idx))
    f_value = (2 * procession * recall) / (procession + recall)
    print("p value：" + str(procession), "r value：" + str(recall), "f value：" + str(f_value))
    # 计算保真度和压缩比
    fidelity, ratio = fidelity_and_ratio(features, keyframes_idx, test_index)
    print("fidelity value：" + str(fidelity), "ratio value：" + str(ratio))
    res = {
        "f_value": f_value,
        "p_value": procession,
        "r_value": recall,
        "fidelity_value": fidelity,
        "ratio_value": ratio
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


def handle_video(dir_path, video_path):
    res_path = os.path.join(dir_path, "res.json")
    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    keyframes_idx = res['keyframes_idx']
    label_path = os.path.join(dir_path, "label.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read()
    label_idx = [int(i) for i in content.split('\n')]
    evaluation(keyframes_idx, label_idx, video_path, dir_path)


def main(file_dir):
    for item in os.listdir(file_dir):
        if item.endswith(".mp4"):
            basename = os.path.splitext(os.path.basename(item))[0]
            dir_path = os.path.join(file_dir, basename)
            video_path = os.path.join(file_dir, item)
            handle_video(dir_path, video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir", type=str, help="Dataset dir")
    args = parser.parse_args()
    main(args.file_dir)