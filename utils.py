import pathlib
from pathlib import Path
import os
import torch
import numpy as np
import random
import warnings
import torch

# config for wandb
def config(args):
    warnings.filterwarnings("ignore", category=UserWarning)

    #     OUTPUT_DIR: Path("./output")
    #     LOGS_DIR: Path(OUTPUT_DIR, "logs")
    #     MODEL_DIR: Path(OUTPUT_DIR, "models")
    #     LOGS_DIR.mkdir(parents=True, exist_ok=True)
    #     MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # }

# Seeding and reproducibility
def seed_all(seed: int = 42):
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

def get_stacked_tensor(list_of_list_of_tensors):
    """Converts a list of list of tensors to a pytorch tensor"""
    stacked_tensor = torch.stack([torch.stack(sublist, dim=0) for sublist in list_of_list_of_tensors], dim=0)
    return stacked_tensor

def convert_list_to_tensor(list_of_tensors):
    """Convert a list of tensors to a 2D tensor"""
    list_of_tensors = torch.stack(list_of_tensors)
    return list_of_tensors

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

