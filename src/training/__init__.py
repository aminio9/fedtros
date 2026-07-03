__all__ = ["run_smoke_test", "run_training"]


def __getattr__(name: str):
    if name == "run_training":
        from src.training.centralized import run_training

        return run_training
    if name == "run_smoke_test":
        from src.training.smoke import run_smoke_test

        return run_smoke_test
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")
