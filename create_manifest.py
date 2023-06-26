import csv
import pandas as pd
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
    


if __name__ == '__main__':
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
