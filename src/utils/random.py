import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = False):
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    Args:
        seed (int): The seed value to set.
        deterministic (bool, optional): Whether to set torch to deterministic mode. Defaults to True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True