"""Data loading and preprocessing utilities."""

<<<<<<< HEAD
from src.data.io import load_tensor_dataset
from src.data.preprocessing import run_preprocessing

__all__ = ["load_tensor_dataset", "run_preprocessing"]
=======
from src.data.loading import load_tensor_dataset
from src.data.preprocessing import run_preprocessing
from src.data.splits import dirichlet_split

__all__ = ["dirichlet_split", "load_tensor_dataset", "run_preprocessing"]
>>>>>>> ea28efe (Initial commit with updated source code)
