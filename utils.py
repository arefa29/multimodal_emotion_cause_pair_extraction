import pathlib
import os
import torch
import numpy as np
import random

# config for wandb
class config:
    warnings.filterwarnings("ignore", category=UserWarning)

    DATA_DIR = Path("./data")
    OUTPUT_DIR = Path("./")
    LOGS_DIR = Path(OUTPUT_DIR, "logs")
    MODEL_DIR = Path(OUTPUT_DIR, "models")
    OOF_DIR = Path(OUTPUT_DIR, "oof")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True) # out-of-fold validation

# Seeding and reproducibility
def seed_all(seed: int = 1992):
    """Seed all random number generators"""
    print("Using Seed Number {}".format(seed));

    os.environ["PYTHONHASHSEED"] = str(seed) # set PYTHONHASHSEED end var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed) #pytorch both CPU and CUDA
    np.random.seed(seed) # for numpy pseudo-random generators
    random.seed(seed) # for python built-in pseudo-random generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled  = True

def seed_worker(_worker_id):
    """Seed a worker with the given id"""
    worker_seed = torrch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

