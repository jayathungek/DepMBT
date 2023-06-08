import csv
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

def collect_filenames(path: str, ext: str):
    return list(Path(path).glob(f"*.{ext}"))


def create_index_mapping_table(mappings_file):
    mappings = {}
    with open(mappings_file, "r") as fh:
        reader = csv.reader(fh, delimiter=",", quotechar='"')
        for row in reader:
            key = row[1]
            value = row[0]
            mappings[key] = value
        
    return mappings


def write_manifest(save_filename: str, manifest_lines: List[str]):
    with open(save_filename, "w") as fh:
        writer = csv.writer(fh)
        for m in manifest_lines:
            split_lines = m.split(",")
            file_id = split_lines[0]
            label = MAPPINGS_LOOKUP[(split_lines[-1].split(",")[0]).strip().replace('"', '')]
            full_path = IN_DIR / f"{file_id}.mp4"
            writer.writerow([f"{str(full_path.resolve())}", f" {label}"])

def get_manifest_lines(f_manifest, valid_file_ids):
    with open(f_manifest, "r") as fh:
        lines = [
            line for line in fh.readlines() 
            if line.split(",")[0] in valid_file_ids
        ]

    return lines

if __name__ == '__main__':
    IN_DIR = Path(sys.argv[1]).resolve()
    outfile = Path(sys.argv[2]).resolve()
    index_mappings = Path(sys.argv[3]).resolve()
    full_manifest = Path(sys.argv[4]).resolve()
    MAPPINGS_LOOKUP = create_index_mapping_table(str(index_mappings))

    all_wavs = tqdm(collect_filenames(str(IN_DIR), "mp4"))
    mfest_lines = get_manifest_lines(str(full_manifest), [w.stem for w in all_wavs])
    print(len(mfest_lines))
    write_manifest(outfile, mfest_lines)
