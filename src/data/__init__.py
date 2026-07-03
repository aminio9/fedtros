"""Data loading and preprocessing utilities."""

from src.data.loading import load_tensor_dataset
from src.data.preprocessing import run_preprocessing
from src.data.splits import dirichlet_split

__all__ = ["dirichlet_split", "load_tensor_dataset", "run_preprocessing"]
