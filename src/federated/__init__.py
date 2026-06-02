"""Flower client, server, strategy, and orchestration components."""

from importlib import import_module

__all__ = ["run_federated_client", "run_federated_server", "run_federated_simulation"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module("src.federated.run")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
