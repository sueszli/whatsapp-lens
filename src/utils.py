import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = -1) -> None:
    if seed == -1:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_current_dir() -> Path:
    try:
        return Path(__file__).parent.absolute()
    except NameError:
        return Path(os.getcwd())


def get_device(disable_mps=False) -> str:
    if torch.backends.mps.is_available() and not disable_mps:
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
