import _bootstrap  # noqa: F401
import shutil

import hydra
from omegaconf import DictConfig

from src.data import run_preprocessing
from src.utils.config import resolve_path
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(cfg, script_name="preprocess.py", with_device=False)
    assert context.tracker is not None
    metadata = run_preprocessing(cfg, project_root=context.project_root)
    tracker = context.tracker
    tracker.write_json("preprocess_metadata.json", metadata)
    processed_dir = resolve_path(context.project_root, cfg.dataset.preprocessing.output_dir)
    for filename in ("client_class_distribution.csv", "partition_manifest.jsonl"):
        source = processed_dir / filename
        if source.exists():
            shutil.copy2(source, tracker.run_dir / filename)


if __name__ == "__main__":
    main()
