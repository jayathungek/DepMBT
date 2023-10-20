import csv
from types import ModuleType
import pandas as pd
import sys
from pathlib import Path
from typing import List, Callable, Tuple

from tqdm import tqdm

from tokenizer import Tokenizer
from constants import *
from datasets import get_vid_duration

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

def append_cols_right(file1, file1_header, file2, file2_header, data_dir, outfile):
    """
    Appends the columns of file2 to the columns of file1
    """
    df1 = pd.read_csv(file1, names=file1_header)
    df2 = pd.read_csv(file2, names=file2_header)
    assert df1.shape[1] == len(file1_header), "Mismatch of column count and header name count in file 1"
    assert df2.shape[1] == len(file2_header), "Mismatch of column count and header name count in file 2"
    assert df1.shape[0] == df2.shape[0], "Mismatch of number of rows between file 1 and 2"
    joined = pd.concat([df1, df2], axis=1)
    joined['filename'] = joined['filename'].map(lambda s: str(data_dir / s.replace("'", "")))
    joined.to_csv(outfile, index=False)
    

class Manifest:
    def __init__(self, 
                 dataset_const_namespace: ModuleType,
                 ) -> None:
        self.path_label_mappings: List[Tuple] # first item in tuple is filepath, rest are labels

        self.constants = dataset_const_namespace
        self.dataset_root = Path(self.constants.DATA_DIR)
        self.tokenizer = Tokenizer(dataset_const_namespace)
    
    def create(self):
        mappings = self.constants.manifest_fn(self.dataset_root)
        self.prune_and_save(mappings)

    def prune_and_save(self, path_label_mappings):
        """
        runs the video files in the manifest through the pre-processing pipeline
        and excludes any failures from the final manifest
        """
        failed = 0
        ok_lines = []
        for row in tqdm(path_label_mappings):
            filepath, *_ = row
            filepath = Path(filepath).resolve()
            try:
                duration = get_vid_duration(filepath)
                rgb, spec = self.tokenizer.make_input(filepath, duration, self.constants.SAMPLING_RATE)
                rgb = rgb.reshape((CHANS * FRAMES, 
                                    HEIGHT, 
                                    WIDTH)
                                ).unsqueeze(0)  # f, c, h, w -> 1, c*f, h, w
                spec = spec.unsqueeze(0)
                row.insert(1, f"{duration:.2f}")
                ok_lines.append(row)
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")
                failed += 1
        
        dest = self.dataset_root / f"{self.constants.NAME}_pruned.csv"
        with open(dest, "w") as fh:
            writer = csv.writer(fh)
            writer.writerows(ok_lines)
            print(f"Wrote {len(ok_lines)} rows to {dest}, {failed} failed.")

if __name__ == '__main__':
    from datasets import enterface, emoreact
    m = Manifest(enterface)
    m.create()
    exit()
    BASE_DIR = "/root/intelpa-1/datasets/EmoReact/EmoReact_V_1.0"
    f1 = Path(BASE_DIR) / "Labels" / sys.argv[1]
    f2 = Path(BASE_DIR) / "Labels" / sys.argv[2]
    o  = Path(BASE_DIR) / "Labels" / sys.argv[3]
    append_cols_right(f1, ["filename"], 
                      f2, ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration", "Valence"],
                      Path(BASE_DIR) / "Data/Validation",
                      o,
                      )
    exit()
    IN_DIR = Path(sys.argv[1]).resolve()
    outfile = Path(sys.argv[2]).resolve()
    index_mappings = Path(sys.argv[3]).resolve()
    full_manifest = Path(sys.argv[4]).resolve()
    MAPPINGS_LOOKUP = create_index_mapping_table(str(index_mappings))

    all_wavs = tqdm(collect_filenames(str(IN_DIR), "mp4"))
    mfest_lines = get_manifest_lines(str(full_manifest), [w.stem for w in all_wavs])
    print(len(mfest_lines))
    write_manifest(outfile, mfest_lines)
