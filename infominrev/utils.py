import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
