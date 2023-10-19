from pathlib import Path

from ffprobe import FFProbe


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

        



if __name__ == "__main__":
    from enterface import DATA_DIR
    perform_diagnostic(Path(DATA_DIR), "avi")