from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from omegaconf import OmegaConf

from src.artifacts.communication import build_communication_metrics
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    metadata: dict[str, Any]
    config: Any
    metrics: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        return _load_json(metrics_path)

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        try:
            frame = pd.read_csv(metrics_csv)
            if not frame.empty:
                return frame.iloc[-1].dropna().to_dict()
        except Exception as exc:
            logger.warning("Could not read %s: %s", metrics_csv, exc)

    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        lines = [line for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except Exception as exc:
                logger.warning("Could not parse last metrics line in %s: %s", metrics_jsonl, exc)
    return {}


def _load_run_summary(run_dir: Path) -> RunSummary:
    metadata = _load_json(run_dir / "metadata.json")
    config = None
    config_path = run_dir / "resolved_config.yaml"
    if config_path.exists():
        try:
            config = OmegaConf.load(config_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", config_path, exc)
    return RunSummary(run_dir=run_dir, metadata=metadata, config=config, metrics=_load_metrics(run_dir))


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        value = OmegaConf.select(cfg, path, default=default)
    except Exception:
        value = default
    return default if value in (None, "???") else value


def _metric_value(metrics: dict[str, Any], candidates: Iterable[str]) -> float | None:
    for candidate in candidates:
        value = metrics.get(candidate)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_existing_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _write_csv(frame: pd.DataFrame, output_path: Path, *, sort_by: list[str] | None = None) -> Path | None:
    if frame.empty:
        return None
    if sort_by:
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info("Saved %s (%d rows)", output_path, len(frame))
    return output_path


def _suite_method(summary: RunSummary) -> str:
    method = _cfg_select(summary.config, "experiment.method")
    if method is None:
        method = summary.metadata.get("method")
    return str(method or "")


def _dataset_name(summary: RunSummary) -> str:
    dataset = _cfg_select(summary.config, "dataset.name")
    if dataset is None:
        dataset = summary.metadata.get("dataset")
    return str(dataset or "")


def _known_labels(summary: RunSummary) -> list[str]:
    labels = _cfg_select(summary.config, "dataset.known_labels")
    if labels is None:
        labels = _cfg_select(summary.config, "dataset.preprocessing.known_labels", [])
    return [str(label) for label in labels or []]


def _source_labels(summary: RunSummary) -> list[str]:
    labels = _cfg_select(summary.config, "dataset.source_labels", [])
    return [str(label) for label in labels or []]


def _class_names_count(summary: RunSummary, project_root: Path) -> int | None:
    output_dir = _cfg_select(summary.config, "dataset.preprocessing.output_dir")
    if not output_dir:
        return None
    class_names_path = resolve_path(project_root, Path(str(output_dir)) / "class_names.json")
    if not class_names_path.exists():
        return None
    try:
        return len(json.loads(class_names_path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Could not read %s: %s", class_names_path, exc)
        return None


def _compute_openness(summary: RunSummary, project_root: Path) -> float | None:
    known_count = len(_known_labels(summary))
    total_count = len(_source_labels(summary))
    if total_count <= 0:
        total_count = _class_names_count(summary, project_root) or 0
    if known_count <= 0 or total_count <= 0 or total_count < known_count:
        return None
    openness = 1.0 - ((2.0 * known_count) / float(known_count + total_count)) ** 0.5
    return float(max(openness, 0.0))


def _final_accuracy(summary: RunSummary) -> float | None:
    return _metric_value(
        summary.metrics,
        ("test/accuracy", "accuracy", "openset_overall_acc", "open_set/overall_acc"),
    )


def _final_macro_f1(summary: RunSummary) -> float | None:
    return _metric_value(
        summary.metrics,
        ("test/macro_f1", "openset_f1_macro", "macro_f1", "f1_macro", "f1"),
    )


def _open_set_auroc(summary: RunSummary) -> float | None:
    return _metric_value(summary.metrics, ("open_set/auroc", "openset_auroc", "auroc"))


def _row_metadata(summary: RunSummary) -> dict[str, Any]:
    return {
        "run_id": summary.metadata.get("run_id", summary.run_dir.name),
        "run_dir": str(summary.run_dir),
        "dataset": _dataset_name(summary),
        "method": _suite_method(summary),
        "seed": summary.metadata.get("seed", _cfg_select(summary.config, "seed")),
    }


def _collect_scalability_rows(summaries: list[RunSummary], project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        num_clients = _cfg_select(summary.config, "federated.num_clients", summary.metadata.get("num_clients"))
        final_accuracy = _final_accuracy(summary)
        if num_clients is None or final_accuracy is None:
            continue
        rows.append(
            {
                **_row_metadata(summary),
                "num_clients": int(num_clients),
                "final_accuracy": float(final_accuracy),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.groupby(["num_clients"], as_index=False).agg(
        {
            "final_accuracy": "mean",
            "dataset": "first",
            "method": "first",
            "seed": "first",
            "run_id": "first",
            "run_dir": "first",
        }
    )


def _collect_openness_rows(summaries: list[RunSummary], project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        openness = _compute_openness(summary, project_root)
        auroc = _open_set_auroc(summary)
        if openness is None or auroc is None:
            continue
        rows.append(
            {
                **_row_metadata(summary),
                "openness": float(openness),
                "auroc": float(auroc),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.groupby(["method", "openness"], as_index=False).agg(
        {
            "auroc": "mean",
            "dataset": "first",
            "seed": "first",
            "run_id": "first",
            "run_dir": "first",
        }
    )


def _collect_cross_dataset_rows(summaries: list[RunSummary]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        dataset = _dataset_name(summary)
        accuracy = _metric_value(summary.metrics, ("test/accuracy", "accuracy"))
        macro_f1 = _final_macro_f1(summary)
        auroc = _open_set_auroc(summary)
        for metric_name, value in (
            ("accuracy", accuracy),
            ("f1", macro_f1),
            ("auroc", auroc),
        ):
            if value is None:
                continue
            rows.append(
                {
                    **_row_metadata(summary),
                    "dataset": dataset,
                    "metric": metric_name,
                    "metric_value": float(value),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.groupby(["dataset", "metric"], as_index=False).agg(
        {
            "metric_value": "mean",
            "method": "first",
            "seed": "first",
            "run_id": "first",
            "run_dir": "first",
        }
    )


def _heterogeneity_label(summary: RunSummary) -> str | None:
    iid = bool(_cfg_select(summary.config, "dataset.preprocessing.iid", False))
    if iid:
        return "IID"
    alpha = _cfg_select(summary.config, "dataset.preprocessing.alpha")
    if alpha is None:
        alpha = summary.metadata.get("alpha")
    if alpha is None:
        return None
    return str(alpha)


def _collect_seed_rows(summaries: list[RunSummary]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        heterogeneity = _heterogeneity_label(summary)
        accuracy = _final_accuracy(summary)
        seed = summary.metadata.get("seed", _cfg_select(summary.config, "seed"))
        if heterogeneity is None or accuracy is None or seed is None:
            continue
        rows.append(
            {
                **_row_metadata(summary),
                "seed": int(seed),
                "heterogeneity": heterogeneity,
                "accuracy": float(accuracy),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.groupby(["seed", "heterogeneity"], as_index=False).agg(
        {
            "accuracy": "mean",
            "dataset": "first",
            "method": "first",
            "run_id": "first",
            "run_dir": "first",
        }
    )


def _collect_ablation_rows(summaries: list[RunSummary]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        macro_f1 = _final_macro_f1(summary)
        configuration = _suite_method(summary)
        if macro_f1 is None or not configuration:
            continue
        rows.append(
            {
                **_row_metadata(summary),
                "configuration": configuration,
                "macro_f1": float(macro_f1),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.groupby(["configuration"], as_index=False).agg(
        {
            "macro_f1": "mean",
            "dataset": "first",
            "method": "first",
            "seed": "first",
            "run_id": "first",
            "run_dir": "first",
        }
    )


def _collect_latent_rows(run_dirs: list[Path], project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "latent_embeddings.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Could not read latent embeddings %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        metadata = _load_json(run_dir / "metadata.json")
        cfg = _load_run_summary(run_dir).config
        frame = frame.copy()
        frame["run_id"] = metadata.get("run_id", run_dir.name)
        frame["run_dir"] = str(run_dir)
        frame["source_run_dir"] = str(run_dir)
        frame["dataset"] = _dataset_name(RunSummary(run_dir, metadata, cfg, {}))
        frame["method"] = _suite_method(RunSummary(run_dir, metadata, cfg, {}))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_communication_rows(run_dirs: list[Path], project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "communication_metrics.csv"
        if path.exists():
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)
                continue
        else:
            frame = build_communication_metrics(run_dir=run_dir, project_root=project_root)
        if frame.empty:
            continue
        summary = _load_run_summary(run_dir)
        frame = frame.copy()
        if "round" not in frame.columns and "logical_round" in frame.columns:
            frame = frame.rename(columns={"logical_round": "round"})
        if "source_run_dir" not in frame.columns:
            frame["source_run_dir"] = str(run_dir)
        frame["run_id"] = summary.metadata.get("run_id", run_dir.name)
        frame["dataset"] = _dataset_name(summary)
        frame["seed"] = summary.metadata.get("seed", _cfg_select(summary.config, "seed"))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if {"method", "round", "cumulative_mb"}.issubset(frame.columns):
        frame = frame.groupby(["method", "round", "cumulative_mb"], as_index=False).agg(
            {
                "accuracy": "mean",
                "run_id": "first",
                "dataset": "first",
                "seed": "first",
                "source_run_dir": "first",
            }
        )
    return frame


def build_suite_artifacts(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Path]:
    summaries = [_load_run_summary(run_dir) for run_dir in run_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    comparison_path = output_dir / "comparison_metrics.csv"
    if comparison_path.exists():
        generated["comparison_metrics.csv"] = comparison_path

    frames_and_paths = [
        ("scalability.csv", _collect_scalability_rows(summaries, project_root), ["num_clients"]),
        ("openness_metrics.csv", _collect_openness_rows(summaries, project_root), ["openness"]),
        ("cross_dataset_metrics.csv", _collect_cross_dataset_rows(summaries), ["dataset", "metric"]),
        ("seed_robustness.csv", _collect_seed_rows(summaries), ["seed", "heterogeneity"]),
        ("ablation_metrics.csv", _collect_ablation_rows(summaries), ["configuration"]),
    ]
    for filename, frame, sort_by in frames_and_paths:
        path = _write_csv(frame, output_dir / filename, sort_by=sort_by)
        if path is not None:
            generated[filename] = path

    latent_frame = _collect_latent_rows(run_dirs, project_root)
    latent_path = _write_csv(latent_frame, output_dir / "latent_embeddings.csv")
    if latent_path is not None:
        generated["latent_embeddings.csv"] = latent_path

    communication_frame = _collect_communication_rows(run_dirs, project_root)
    communication_path = _write_csv(
        communication_frame,
        output_dir / "communication_metrics.csv",
        sort_by=["method", "round"] if {"method", "round"}.issubset(communication_frame.columns) else None,
    )
    if communication_path is not None:
        generated["communication_metrics.csv"] = communication_path

    manifest = {
        "input_runs": [str(run_dir) for run_dir in run_dirs],
        "generated_files": {name: str(path) for name, path in sorted(generated.items())},
    }
    (output_dir / "suite_artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Saved suite artifacts manifest to %s", output_dir / "suite_artifacts_manifest.json")
    generated["suite_artifacts_manifest.json"] = output_dir / "suite_artifacts_manifest.json"
    return generated
