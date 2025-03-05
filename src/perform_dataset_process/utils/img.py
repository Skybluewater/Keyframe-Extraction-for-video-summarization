import os
import json
import re
from bisect import bisect_left, bisect_right
import logging
import base64
import warnings

log = logging.getLogger(__name__)

warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use the updated video processing utilities.",
    DeprecationWarning,
    stacklevel=2
)
def align_img_with_chunk(image_file_path, chunk_file_path=None, **kwargs):
    # Load images with timestamps
    pattern = re.compile(r"keyframe_(\d+\.\d+)\.jpg")
    images = sorted(
        [
            (float(match.group(1)), os.path.join(image_file_path, file))
            for file in os.listdir(image_file_path)
            if (match := pattern.match(file))
        ],
        key=lambda x: x[0],
    )
    log.info(f"Found {len(images)} key frames.")
    if not images:
        return []

    # Load chunk file (default to "chunk_4.json" in image_file_path if not provided)
    if chunk_file_path is None:
        chunk_file_path = os.path.join(image_file_path, "chunk_4.json")
    with open(chunk_file_path, "r", encoding="utf-8") as f:
        chunk_file = json.load(f)
        chunks = chunk_file["chunks"]
    if not chunks:
        return []

    # Precompute chunk midpoints, starts, and ends (assume sorted chunks)
    chunk_starts = [c["start"] for c in chunks]
    chunk_ends = [c["end"] for c in chunks]

    aligned_results = []
    for ts, img_path in images:
        # Use binary search to find closest chunk by midpoint
        pos = bisect_left(chunk_starts, ts)
        closest = (
            0
            if pos == 0
            else (
                len(chunk_starts) - 1
                if pos == len(chunk_starts)
                else (
                    pos
                    if abs(chunk_starts[pos] - ts) < abs(ts - chunk_starts[pos - 1])
                    else pos - 1
                )
            )
        )

        # Validate overlap: if image timestamp not within chunk range, set to None
        closest_chunk = (
            chunks[closest].copy()
            if chunks[closest]["start"] <= ts <= chunks[closest]["end"]
            else None
        )

        # Find previous chunk (if within 5s before)
        pos_end = bisect_right(chunk_ends, ts) - 1
        prev_chunk = (
            chunks[pos_end].copy()
            if (pos_end >= 0 and abs(ts - chunk_ends[pos_end]) <= 5)
            else None
        )

        # Find next chunk (if within 5s after)
        pos_start = bisect_left(chunk_starts, ts)
        next_chunk = None  # chunks[pos_start].copy() if (pos_start < len(chunks) and abs(chunk_starts[pos_start] - ts) <= 5) else None

        aligned_results.append(
            {
                "image": img_path,
                "closest_chunk": closest_chunk,
                "prev_chunk": prev_chunk,
                "next_chunk": next_chunk,
            }
        )

    # Write output if requested
    output = {
        "key": chunk_file.get("key", ""),
        "text": chunk_file.get("text", ""),
        "aligned": aligned_results,
    }
    output_path = kwargs.get("output_path") or os.path.join(
        image_file_path, "aligned_5.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    return output


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


if __name__ == "__main__":
    align_img_with_chunk("./output_frames")
