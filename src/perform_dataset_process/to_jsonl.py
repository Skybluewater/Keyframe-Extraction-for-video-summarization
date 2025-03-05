import os
import argparse
import json
import jsonlines
import logging

from utils import image_to_base64, format_vlm_message, format_llm_message
from PROMPT import PROMPTS
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


"""
    I want to extract the data from the specific dir to jsonl
"""

def read_vlm_data(save_dir):
    dir_name = os.path.basename(save_dir)
    aligned_file_name = os.path.join(save_dir, "aligned_5.json")
    with open(aligned_file_name, "r", encoding="utf-8") as f:
        aligned_file = json.load(f)
        bg = aligned_file['key']
        aligned_results = aligned_file['aligned']
        system_prompt = PROMPTS["system"][0]
        content_prompt = PROMPTS["frame_expansion"][2]
        ret: list[dict] = []
        for aligned_result in aligned_results:
            c_c = aligned_result["closest_chunk"]
            p_c = aligned_result["prev_chunk"]
            if c_c is None and p_c is None:
                continue
            dp = p_c['text'] if c_c is None else c_c['text'] if p_c is None else p_c['text'] + c_c['text']
            image_path = aligned_result["image"]
            base64_image = image_to_base64(image_path)
            text = content_prompt.format(background=bg, description=dp)
            vlm_dict = format_vlm_message(system_prompt, base64_image, text, id=aligned_result['image'])
            ret.append(vlm_dict)
    log.info(f"Format {len(ret)} VLM data in {dir_name}")
    return ret


def save_data_to_jsonl(data, save_dir):
    log.info(f"Saving jsonl file to {save_dir}/data.jsonl")
    with open(os.path.join(save_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        writer = jsonlines.Writer(f)
        for d in data:
            writer.write(d)
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image files to JSONL format")
    parser.add_argument("--file_dir", type=str, required=True, help="File dir")
    args = parser.parse_args()
    file_dir = args.file_dir
    jsonl_content: list[dict] = []
    for dir_names in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, dir_names)):
            log.info(f"Reading data from {dir_names}")
            save_dir = os.path.join(file_dir, dir_names)
            jsonl_content.extend(read_vlm_data(save_dir))
    save_data_to_jsonl(jsonl_content, file_dir)