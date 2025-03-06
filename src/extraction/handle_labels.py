import os
import re

def extract_labels(dir):
    pattern = re.compile(r"(\d+).jpg")
    labels = []
    for item in os.listdir(dir):
        match = pattern.match(item)
        if match:
            labels.append(int(match.group(1)))
    labels.sort()
    with open(os.path.join(dir, "label.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def main(file_dir):
    for item in os.listdir(file_dir):
        dir = os.path.join(file_dir, item)
        if os.path.isdir(dir):
            extract_labels(dir)


if __name__ == "__main__":
    main("./Keyframe-extraction-main/Dataset/Keyframe")