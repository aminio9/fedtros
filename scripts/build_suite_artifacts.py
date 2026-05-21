import hashlib
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.artifacts.suite import build_suite_artifacts
from src.evaluation.compare import compare_runs
from src.tracking import initialize_run


@hydra.main(config_path="../src/configs", config_name="suite_artifacts", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    run_dirs = [Path(run) for run in cfg.runs]
    if not run_dirs:
        raise ValueError("Provide runs=[outputs/run1,outputs/run2,...] to build_suite_artifacts.py.")

    if str(cfg.tracking.run_id) == "suite_artifacts":
        resolved_runs = [
            str(run if run.is_absolute() else (project_root / run).resolve())
            for run in run_dirs
        ]
        digest = hashlib.sha1("|".join(sorted(resolved_runs)).encode("utf-8")).hexdigest()[:12]
        cfg.tracking.run_id = f"suite_artifacts_{digest}"

    tracker = initialize_run(cfg, project_root=project_root, script_name="build_suite_artifacts.py")

    compare_runs(cfg, project_root=project_root)
    generated = build_suite_artifacts(
        run_dirs=[run if run.is_absolute() else project_root / run for run in run_dirs],
        output_dir=tracker.run_dir,
        project_root=project_root,
    )
    tracker.write_json(
        "suite_artifacts.json",
        {
            "input_runs": [str(run) for run in run_dirs],
            "generated_files": {name: str(path) for name, path in sorted(generated.items())},
        },
    )


if __name__ == "__main__":
    main()
