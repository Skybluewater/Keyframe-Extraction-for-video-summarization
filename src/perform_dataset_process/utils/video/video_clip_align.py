import os
import json
import re
from bisect import bisect_left, bisect_right
import logging
import base64


log = logging.getLogger(__name__)


def align_clip_with_chunk(video_file_path, **kwargs):
    video_name = os.path.basename(video_file_path)
    basename = os.path.splitext(video_name)[0]
    dir_path = os.path.join(os.path.dirname(video_file_path), basename)
    
    # open scenes
    scenes_path = os.path.join(dir_path, "scenes.json")
    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)['scene']

    # Load chunk file (default to "chunk_4.json" in image_file_path if not provided)
    chunk_file_path = os.path.join(dir_path, "chunk_4.json")
    with open(chunk_file_path, "r", encoding="utf-8") as f:
        chunk_file = json.load(f)
        chunks = chunk_file["chunks"]
    if not chunks:
        return []

    # Precompute chunk midpoints, starts, and ends (assume sorted chunks)
    chunk_starts = [c["start"] for c in chunks]
    chunk_ends = [c["end"] for c in chunks]

    aligned_results = []
    for item in scenes:
        startframe = item['start_frame']
        endframe = item['end_frame']
        starttime = item['start_time']
        endtime = item['end_time']
        
        # rough match
        # match only the clip that is fully contained in the chunk
        head_pos_r = bisect_right(chunk_ends, starttime)
        tail_pos_r = bisect_left(chunk_starts, endtime)
        
        # exact match
        # match the clip that has any overlap with the chunk
        head_pos_e = bisect_left(chunk_starts, starttime)
        tail_pos_e = bisect_right(chunk_ends, endtime)
        
        # save chunks
        rough_chunk = None
        exact_chunk = None
        exact_chunk_text = ""
        rough_chunk_text = ""
        if head_pos_r < tail_pos_r:
            rough_chunk = chunks[head_pos_r:tail_pos_r]
            rough_chunk_text = " ".join([c["text"] for c in rough_chunk]).strip()
        if head_pos_e < tail_pos_e:
            exact_chunk = chunks[head_pos_e:tail_pos_e]
            exact_chunk_text = " ".join([c["text"] for c in exact_chunk]).strip()

        aligned_results.append(
            {
                "clip": {
                    "start_frame": startframe,
                    "end_frame": endframe,
                    "start_time": starttime,
                    "end_time": endtime,
                },
                "exact_chunk": {
                    "chunks": exact_chunk,
                    "text": exact_chunk_text,
                },
                "rough_chunk": {
                    "chunks": rough_chunk,
                    "text": rough_chunk_text,
                },
            }
        )

    # Write output if requested
    output = {
        "key": chunk_file.get("key", ""),
        "text": chunk_file.get("text", ""),
        "aligned": aligned_results,
    }
    output_path = kwargs.get("output_path") or os.path.join(
        dir_path, "clips.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    return output


if __name__ == "__main__":
    align_clip_with_chunk("./test/test.mp4")