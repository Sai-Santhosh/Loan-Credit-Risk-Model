"""
Helper Utilities - Common utility functions.

Provides:
- Random seed management
- Memory profiling
- Timing decorators
- YAML utilities
"""

import functools
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import yaml

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"Random seed set to {seed}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage in MB
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "percent": process.memory_percent(),
    }


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with timer("Training"):
        ...     model.fit(X, y)
    """
    start_time = time.time()
    logger.info(f"Starting: {name}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed: {name} ({elapsed:.2f}s)")


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Example:
        >>> @timed
        ... def train_model():
        ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    
    return wrapper


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        path: Path to YAML file
    
    Returns:
        Parsed YAML content
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        path: Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Key separator
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_dataframe(df, chunk_size: int = 10000):
    """
    Generator to chunk DataFrame.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
    
    Yields:
        DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


__all__ = [
    "set_seed",
    "get_memory_usage",
    "timer",
    "timed",
    "load_yaml",
    "save_yaml",
    "ensure_dir",
    "flatten_dict",
    "chunk_dataframe",
]
