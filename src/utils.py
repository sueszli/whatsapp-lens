import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = -1) -> None:
    if seed == -1:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)


def get_current_dir() -> Path:
    try:
        return Path(__file__).parent.absolute()
    except NameError:
        return Path(os.getcwd())
