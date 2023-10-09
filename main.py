import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import os
os.path.insert("./utils")
from utils import utils

@dataclass
class FilePaths:
    """Class to keep track of the files"""

    train_videos: pathlib.Path = pathlib.Path(config.DATA_DIR, "train")
    test_videos: pathlib.Path = pathlib.Path(config.DATA_DIR, "test")
    valid_videos: pathlib.Path = pathlib.Path(config.DATA_DIR, "valid")

    train_text: pathlib.Path = pathlib.Path(config.DATA_DIR, "text")
    weight_path: pathlib.Path = pathlib.Path(config.MODEL_DIR)
    oof_csv: pathlib.Path = pathlib.Path(config.OOF_DIR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class DataLoaderParams:
    """Class to keep track of dataloader parameters"""

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )
