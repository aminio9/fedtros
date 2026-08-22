"""Study definitions, matrix expansion, dry-run status, and paired-partition policy.

The study layer is the research-facing execution contract.  It expands only genuine
experimental dimensions (seed, alpha, clients, holdout, variant) while leaving fixed
scientific settings in YAML study/method configs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.infrastructure.run_id import generate_run_id

CANONICAL_SEEDS = (17, 42, 73, 101, 137)

# Profiles change engineering budget only.  paper_final/reproduction never shorten a study.
STAGE_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "federated.num_rounds": 2,
        "training.local_epochs": 1,
        "federated.num_clients": 2,
        "dataset.preprocessing.num_clients": 2,
        "dataset.preprocessing.smoke": True,
        "dataset.preprocessing.smoke_min_samples_per_client": 1,
        "dataset.preprocessing.smoke_max_samples_per_class": 512,
        "open_set.calibration.min_samples_per_class": 2,
        "open_set.prototype_rank.prototype.num_prototypes_per_class": 2,
        "open_set.prototype_rank.prototype.min_samples_per_prototype": 2,
        "open_set.prototype_rank.prototype.negative.num_prototypes": 4,
        "open_set.prototype_rank.prototype.negative.max_samples": 128,
        "open_set.evaluate_each_round": True,
        "open_set.evaluate_every_n_rounds": 1,
        "runtime.allow_cpu_fallback": True,
    },
    "development": {"federated.num_rounds": 5, "training.local_epochs": 1},
    "tuning": {},
    "ablation": {},
    "paper_final": {},
    "reproduction": {},
}


@dataclass
class PlannedRun:
    study_id: str
    stage: str
    dataset: str
    method: str
    alpha: float
    iid: bool
    unknown_labels: list[str]
    seed: int
    num_clients: int
    variant: str
    run_id: str
    human_name: str
    config_hash: str
    partition_file: str
    overrides: dict[str, Any] = field(default_factory=dict)
    status: str = "NEW"


@dataclass
class DryRunSummary:
    study_id: str
    stage: str
    total_runs: int = 0
    completed_runs: int = 0
    missing_runs: int = 0
    failed_runs: int = 0
    interrupted_runs: int = 0
    resumable_runs: int = 0
    new_runs: int = 0
    planned_runs: list[PlannedRun] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 108)
        print(f" STUDY EXECUTION PLAN: {self.study_id} | stage={self.stage}")
        print("=" * 108)
        print(
            f" expected={self.total_runs} completed={self.completed_runs} "
            f"missing={self.missing_runs} failed={self.failed_runs} resumable={self.resumable_runs}"
        )
        print("-" * 108)
        print(f"{'#':<3} {'Status':<12} {'Method':<13} {'Dataset':<10} {'a':<6} {'C':<4} {'Seed':<5} {'Variant':<20} Run ID")
        print("-" * 108)
        for idx, run in enumerate(self.planned_runs, 1):
            print(
                f"{idx:<3} {run.status:<12} {run.method:<13} {run.dataset:<10} "
                f"{run.alpha:<6g} {run.num_clients:<4} {run.seed:<5} {run.variant:<20} {run.run_id}"
            )
        print("=" * 108 + "\n")


def get_paired_partition_path(
    project_root: Path,
    dataset: str,
    alpha: float,
    seed: int,
    *,
    iid: bool = False,
    num_clients: int = 10,
    known_labels: list[str] | None = None,
    unknown_labels: list[str] | None = None,
    stage: str | None = None,
    partition_profile: str | None = None,
) -> Path:
    """Partition key shared across matched methods for one *scientific condition*.

    The known/unknown protocol is part of the partition identity.  Without it, a
    closed-set E3 partition and an E4/E8 holdout partition could accidentally reuse
    the same relative indices even though their known-training populations differ.
    """
    stage_token = str(stage or "").lower()
    part_dir = project_root / "partitions"
    if stage_token == "smoke":
        part_dir = part_dir / "smoke"
    part_dir = part_dir / dataset.lower()
    known = [str(x) for x in (known_labels or [])]
    unknown = [str(x) for x in (unknown_labels or [])]
    protocol_payload = json.dumps(
        {"known": known, "unknown": unknown}, sort_keys=True, separators=(",", ":")
    )
    protocol_hash = hashlib.sha256(protocol_payload.encode("utf-8")).hexdigest()[:8]
    if unknown:
        label_token = "unk-" + "-".join(x.lower().replace(" ", "-") for x in unknown)
    else:
        label_token = "closed"
    if iid:
        filename = f"iid_{label_token}_{protocol_hash}_c{num_clients}_seed_{seed}.json"
    else:
        filename = (
            f"alpha_{str(alpha).replace('.', 'p')}_{label_token}_{protocol_hash}"
            f"_c{num_clients}_seed_{seed}.json"
        )
    return part_dir / filename


def load_study_config(study_path_or_name: str | Path, project_root: Path) -> dict[str, Any]:
    p = Path(study_path_or_name)
    if p.exists() and p.is_file():
        data = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        return dict(data or {})
    target = str(study_path_or_name).strip()
    sdir = project_root / "src" / "configs" / "study"
    if not sdir.exists():
        raise FileNotFoundError(f"Study directory not found: {sdir}")
    normalized = target.lower().replace("-", "_")
    for f in sorted(sdir.glob("*.yaml")):
        if f.stem.lower() == normalized or f.stem.lower().startswith(normalized.split("_")[0] + "_"):
            data = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
            if str((data or {}).get("study_id", "")).upper() == target.upper() or f.stem.lower() == normalized:
                return dict(data or {})
    for f in sorted(sdir.glob("*.yaml")):
        data = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
        if isinstance(data, dict) and str(data.get("study_id", "")).upper() == target.upper():
            return data
    raise FileNotFoundError(f"Study config not found for {target!r} in {sdir}")


def _variant_entries(study_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = study_cfg.get("variants") or [{"name": "canonical", "overrides": {}}]
    out: list[dict[str, Any]] = []
    for item in variants:
        if isinstance(item, str):
            out.append({"name": item, "overrides": {}})
        else:
            out.append({"name": str(item.get("name", "variant")), "overrides": dict(item.get("overrides", {}))})
    return out


def _known_for_unknown(study_cfg: dict[str, Any], unknowns: list[str]) -> list[str] | None:
    mapping = study_cfg.get("known_labels_by_unknown") or {}
    if len(unknowns) == 1 and unknowns[0] in mapping:
        return list(mapping[unknowns[0]])
    fixed = study_cfg.get("known_labels")
    return list(fixed) if fixed else None


def expand_study_matrix(
    study_cfg: dict[str, Any], *, stage: str = "development",
    seeds: list[int] | tuple[int, ...] | None = None,
    clients: list[int] | tuple[int, ...] | None = None,
    project_root: Path | None = None,
) -> list[PlannedRun]:
    root = project_root or Path(".")
    study_id = str(study_cfg.get("study_id", "E0-VERIFY"))
    methods = list(study_cfg.get("methods", ["fedtros_pr"]))
    datasets = list(study_cfg.get("datasets", ["bnat"]))
    alphas = [float(v) for v in study_cfg.get("alphas", [0.5])]
    iids = [bool(v) for v in study_cfg.get("iids", [False])]
    unknown_sets = [list(v) for v in study_cfg.get("unknown_label_sets", [[]])]
    unknown_by_dataset = {str(k).lower(): list(v) for k, v in (study_cfg.get("unknown_labels_by_dataset") or {}).items()}
    known_by_dataset = {str(k).lower(): list(v) for k, v in (study_cfg.get("known_labels_by_dataset") or {}).items()}
    seed_values = list(seeds) if seeds is not None else list(study_cfg.get("seeds", CANONICAL_SEEDS))
    if clients is not None:
        client_values = [int(v) for v in clients]
    elif stage == "smoke":
        # Smoke validation is an engineering analogue, never the paper-scale client
        # matrix.  Studies may opt into several tiny client counts (E6); otherwise
        # the global smoke profile supplies the canonical two-client setting.
        smoke_clients = study_cfg.get("smoke_num_clients_values")
        if smoke_clients is None:
            smoke_clients = [STAGE_PROFILES["smoke"]["federated.num_clients"]]
        client_values = [int(v) for v in smoke_clients]
    else:
        client_values = [int(v) for v in study_cfg.get("num_clients_values", [study_cfg.get("num_clients", 10)])]
    variants = _variant_entries(study_cfg)
    base = dict(study_cfg.get("base_overrides", {}))
    stage_overrides = STAGE_PROFILES.get(stage, {})

    planned: list[PlannedRun] = []
    for dataset, method, alpha, iid, seed, clients, variant in itertools.product(
        datasets, methods, alphas, iids, seed_values, client_values, variants
    ):
        dataset_unknown_sets = [unknown_by_dataset[str(dataset).lower()]] if str(dataset).lower() in unknown_by_dataset else unknown_sets
        for unknowns in dataset_unknown_sets:
            open_set = bool(unknowns)
            overrides: dict[str, Any] = {
                **base, **stage_overrides, **variant["overrides"],
                "experiment.id": study_id,
                "experiment.method": "FedTROS-PR" if method in {"fedtros", "fedtros_pr"} else method,
                "method": "fedtros_pr" if method in {"fedtros", "fedtros_pr"} else method,
                "dataset": dataset,
                "dataset.preprocessing.alpha": alpha,
                "dataset.preprocessing.iid": iid,
                "dataset.preprocessing.unknown_labels": unknowns,
                "dataset.preprocessing.num_clients": clients,
                "federated.num_clients": clients,
                "seed": int(seed),
                "stage": stage,
                "evaluation.mode": "open_set" if open_set else "closed_set",
                "open_set.enabled": open_set,
                "open_set.method": "prototype_rank" if open_set else "disabled",
                "experiment.variant": variant["name"],
            }
            known = known_by_dataset.get(str(dataset).lower())
            if known is None:
                known = _known_for_unknown(study_cfg, unknowns)
            if known is not None:
                overrides["dataset.preprocessing.known_labels"] = known
                overrides["model.num_classes"] = len(known)

            if method in {"fedtros", "fedtros_pr"}:
                overrides["federated.strategy.name"] = "fedtros_pr"
                if open_set:
                    overrides["open_set.detector"] = "prototype_rank"
                    overrides["open_set.prototype_rank.proser.enabled"] = False
                    overrides["open_set.prototype_rank.energy.train_margin_enabled"] = False
            elif method in {"fedavg", "fedprox"}:
                overrides["federated.strategy.name"] = method

            part = get_paired_partition_path(
                root, dataset, alpha, int(seed), iid=iid, num_clients=clients,
                known_labels=known, unknown_labels=unknowns, stage=stage,
                partition_profile=(
                    hashlib.sha256(
                        json.dumps(
                            {
                                key: value
                                for key, value in overrides.items()
                                if key.startswith("dataset.preprocessing.smoke_")
                            },
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()[:8]
                    if stage == "smoke"
                    else None
                ),
            )
            overrides["dataset.partition_file"] = part.as_posix()
            run_id, human_name, config_hash = generate_run_id(overrides, study_id=study_id)
            planned.append(PlannedRun(
                study_id=study_id, stage=stage, dataset=dataset, method=str(method), alpha=alpha,
                iid=iid, unknown_labels=unknowns, seed=int(seed), num_clients=clients,
                variant=variant["name"], run_id=run_id, human_name=human_name,
                config_hash=config_hash, partition_file=str(part), overrides=overrides,
            ))
    return planned


def _manifest(run_dir: Path) -> dict[str, Any]:
    for p in (run_dir / "metadata" / "run_manifest.json", run_dir / "run_manifest.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {"status": "CORRUPTED"}
    return {}


def perform_dry_run(planned_runs: list[PlannedRun], *, output_base_dir: Path) -> DryRunSummary:
    summary = DryRunSummary(
        study_id=planned_runs[0].study_id if planned_runs else "E0-VERIFY",
        stage=planned_runs[0].stage if planned_runs else "development",
        total_runs=len(planned_runs), planned_runs=planned_runs,
    )
    for run in planned_runs:
        run_dir = output_base_dir / "runs" / run.run_id
        if not run_dir.exists():
            run.status = "NEW"; summary.new_runs += 1; summary.missing_runs += 1; continue
        data = _manifest(run_dir)
        status = str(data.get("status", "UNINITIALIZED")).upper()
        run.status = status
        if status == "COMPLETED": summary.completed_runs += 1
        elif status == "FAILED": summary.failed_runs += 1; summary.missing_runs += 1
        elif status in {"INTERRUPTED", "RESUMED"}: summary.interrupted_runs += 1; summary.resumable_runs += 1; summary.missing_runs += 1
        else: summary.missing_runs += 1
    return summary


def filter_missing_runs(planned_runs: list[PlannedRun], *, output_base_dir: Path) -> list[PlannedRun]:
    missing: list[PlannedRun] = []
    for run in planned_runs:
        data = _manifest(output_base_dir / "runs" / run.run_id)
        # The study planner does not fully compose Hydra defaults, so the authoritative
        # scientific hash is produced by scripts/run.py.  Run identity is stable across
        # planning/execution; collision validation in run.py protects against config drift.
        if str(data.get("status", "")).upper() == "COMPLETED":
            continue
        missing.append(run)
    return missing
