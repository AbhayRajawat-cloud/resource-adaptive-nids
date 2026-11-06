"""Seed configuration and deterministic execution utilities."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard library and NumPy for reproducibility.
    PyTorch seeding is applied when torch is available (model training stages).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not required for data pipeline stages


def get_device(preference: str = "auto"):
    """Resolve device: CUDA (Kaggle GPU), MPS (Apple Silicon M2), or CPU.
    Returns a torch.device if torch is available, otherwise returns the preference string.
    """
    try:
        import torch
        if preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if preference == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return preference  # Return string preference when torch unavailable

