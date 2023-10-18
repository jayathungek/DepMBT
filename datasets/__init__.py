import math
from typing import Tuple, List
from pathlib import Path

from ffprobe import FFProbe

# first item in tuple is filepath, rest are labels
def enterface_manifest_fn(dataset_root: Path) -> List[Tuple]:
    mapping = {
        "an": 0,
        "sa": 1,
        "di": 2,
        "su": 3,
        "ha": 4,
        "fe": 5
    }
    
    mappings = []
    for p in dataset_root.rglob("*.avi"):
        label = mapping[p.stem.split("_")[1]]
        full_path = p.resolve()
        mappings.append((full_path, label))

    return mappings

def perform_diagnostic(dataset_root: Path, file_ext: str="avi"):
    shortest_vid, shortest_vid_len = None, float("inf")
    longest_vid, longest_vid_len = None, 0
    for p in dataset_root.rglob(f"*.{file_ext}"):
        metadata = FFProbe(str(p))
        for stream in metadata.streams:
            if stream.is_video():
                l = stream.duration_seconds()
                if l >= longest_vid_len:
                    longest_vid_len = l
                    longest_vid = p.name
                if l <= shortest_vid_len:
                    shortest_vid_len = l
                    shortest_vid = p.name
            print(f"{p.name} {l}")
                

    print(f"Longest video: {longest_vid} {longest_vid_len}s\nShortest Video: {shortest_vid} {shortest_vid_len}s")

        

def emoreact_manifest_fn(dataset_root: Path) -> List[Tuple]:
    pass


if __name__ == "__main__":
    from enterface import DATA_DIR
    perform_diagnostic(Path(DATA_DIR), "avi")