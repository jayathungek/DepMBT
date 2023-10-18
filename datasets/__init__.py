from typing import Tuple, List
from pathlib import Path


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
        

def emoreact_manifest_fn(dataset_root: Path) -> List[Tuple]:
    pass


if __name__ == "__main__":
    from enterface import DATA_DIR
    enterface_manifest_fn(Path(DATA_DIR))