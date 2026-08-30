"""Deterministic FedTROS-PR run identity and scientific configuration hashing."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


class RunCollisionError(RuntimeError):
    pass


SCIENTIFIC_CONFIG_KEYS = (
    "seed", "stage", "experiment.id", "experiment.method", "experiment.variant",
    "dataset.name", "dataset.preprocessing.known_labels", "dataset.preprocessing.unknown_labels",
    "dataset.preprocessing.alpha", "dataset.preprocessing.iid", "dataset.preprocessing.num_clients",
    "dataset.preprocessing.val_ratio", "dataset.preprocessing.test_ratio",
    "federated.strategy.name", "federated.num_rounds", "federated.num_clients", "federated.fraction_fit",
    "training.teacher_enabled", "training.teacher_stochastic_training", "training.teacher_beta_kl",
    "training.learning_rate", "training.teacher_lr", "training.student_lr",
    "training.fedtros_global_anchor_weight", "training.fedtros_global_anchor_min_weight",
    "training.fedtros_global_anchor_coverage_power", "training.kd_enabled", "training.kd_gating_enabled",
    "training.alignment_enabled", "training.lambda_kd", "training.lambda_align", "training.adaptive_transfer_weights",
    "training.fedtros_student_hidden_dims", "training.student_osr_enabled", "training.local_epochs",
    "training.student_epochs", "training.teacher_epochs", "training.batch_size",
    "model.name", "model.latent_dim", "model.hidden_dims",
    "open_set.method", "open_set.enabled", "open_set.detector",
    "open_set.prototype_rank.enabled", "open_set.prototype_rank.score_fusion.method",
    "open_set.prototype_rank.prototype.feature_source", "open_set.prototype_rank.prototype.negative.enabled",
    "open_set.calibration.prototype_fit_fraction", "open_set.calibration.threshold_calibration_fraction",
    "open_set.calibration.target_known_fpr", "evaluation.mode",
)


def _sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


def _dataset_slug(text: str) -> str:
    slug = _sanitize_slug(text)
    aliases = {
        "b_nat": "bnat",
        "b_tat": "btat",
        "cic_ids2017": "cicids2017",
        "cic_ids_2017": "cicids2017",
        "ton_iot": "toniot",
    }
    return aliases.get(slug, slug)


def _select(cfg: DictConfig | dict[str, Any], path: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, path, default=default)
        return default if value is None else value
    # Study planner uses a flat dot-key override dictionary; support it directly.
    if path in cfg:
        return cfg[path]
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return default if current is None else current


def compute_scientific_config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    extracted: dict[str, Any] = {}
    for key in SCIENTIFIC_CONFIG_KEYS:
        value = _select(cfg, key, None)
        if value is not None:
            extracted[key] = value
    payload = json.dumps(extracted, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_run_id(cfg: DictConfig | dict[str, Any], *, study_id: str | None = None) -> tuple[str, str, str]:
    """Generate stable identity plus full scientific config hash.

    The run slug is based on experiment identity dimensions rather than the complete resolved
    config, so a study dry-run and the actual Hydra-composed run resolve to the same directory.
    A separate full config hash protects the directory against incompatible reuse.
    """
    config_hash = compute_scientific_config_hash(cfg)
    study = str(study_id or _select(cfg, "experiment.id", "E0-VERIFY")).upper()
    dataset = _dataset_slug(str(_select(cfg, "dataset.name", _select(cfg, "dataset", "bnat"))))
    method_raw = str(_select(cfg, "experiment.method", _select(cfg, "method", "fedtros_pr")))
    method_token = method_raw.lower().replace("-", "_")
    if method_token in {"fedtros", "fedtros_mc", "fedtros_m_c"}:
        method_slug = "fedtros_mc"
    elif method_token in {"fedtros_pr", "fedtros_pr_legacy"}:
        method_slug = "fedtros_pr"
    else:
        method_slug = _sanitize_slug(method_raw)
    iid = bool(_select(cfg, "dataset.preprocessing.iid", False))
    alpha = float(_select(cfg, "dataset.preprocessing.alpha", 0.5))
    unknowns = list(_select(cfg, "dataset.preprocessing.unknown_labels", []) or [])
    clients = int(_select(cfg, "federated.num_clients", _select(cfg, "dataset.preprocessing.num_clients", 10)))
    seed = int(_select(cfg, "seed", 42))
    variant = _sanitize_slug(str(_select(cfg, "experiment.variant", "canonical")))
    rerun_token = _sanitize_slug(str(_select(cfg, "experiment.rerun_token", "")))

    partition_slug = "iid" if iid else f"a{str(alpha).replace('.', 'p')}"
    if not unknowns:
        unknown_slug, unknown_desc = "closed", "Closed-Set"
    elif len(unknowns) == 1:
        unknown_slug = f"{_sanitize_slug(str(unknowns[0]))}unk"
        unknown_desc = f"{unknowns[0]} unknown"
    else:
        unknown_slug, unknown_desc = f"{len(unknowns)}unk", f"{len(unknowns)} unknowns"

    identity = {
        "study": study, "dataset": dataset, "method": method_slug, "partition": partition_slug,
        "unknown": unknown_slug, "clients": clients, "seed": seed, "variant": variant, "rerun": rerun_token,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:6]
    study_slug = _sanitize_slug(study).replace("_", "")
    variant_slug = "" if variant in {"", "canonical"} else f"_{variant}"
    rerun_slug = f"_rerun_{rerun_token}" if rerun_token else ""
    run_id = f"{study_slug}_{dataset}_{method_slug}_{partition_slug}_{unknown_slug}_c{clients}_s{seed}{variant_slug}{rerun_slug}_{identity_hash}"
    human_method = {"fedtros_mc": "FedTROS-MC", "fedtros_pr": "FedTROS-PR"}.get(method_slug, method_raw)
    human = f"{study} | {dataset.upper()} | {human_method} | {'IID' if iid else f'alpha={alpha}'} | {unknown_desc} | c={clients} | s={seed}"
    if variant_slug:
        human += f" | {variant}"
    if rerun_token:
        human += f" | rerun={rerun_token}"
    return run_id, human, config_hash


def validate_run_collision(run_dir: Path, current_config_hash: str) -> None:
    for manifest_path in (run_dir / "metadata" / "run_manifest.json", run_dir / "run_manifest.json"):
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        existing = str(manifest.get("config_hash", ""))
        if existing and existing != current_config_hash:
            raise RunCollisionError(
                f"Run identity {run_dir.name!r} already exists with a different scientific config hash "
                f"({existing[:12]} != {current_config_hash[:12]}). Change the study variant/freeze instead of overwriting."
            )
        return
