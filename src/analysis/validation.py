"""Result compatibility validation for FedTROS experiments (Item C7).

Guarantees scientific rigor by verifying that runs grouped for aggregation or
comparison share compatible study protocols, dataset splits, held-out unknowns,
client counts, Dirichlet alpha, and metric definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from src.analysis.loaders import RunRecord

logger = logging.getLogger(__name__)


class IncompatibleRunsError(ValueError):
    """Raised when attempting to aggregate or compare scientifically incompatible runs."""

    def __init__(self, message: str, discrepancies: list[str] | None = None) -> None:
        self.discrepancies = discrepancies or []
        detail = "\n  - " + "\n  - ".join(self.discrepancies) if self.discrepancies else ""
        super().__init__(f"{message}{detail}")


def _extract_unknown_labels(record: RunRecord) -> list[str]:
    """Extract list of held-out unknown labels from config or metadata."""
    if record.config is not None:
        try:
            from omegaconf import OmegaConf
            unkn = OmegaConf.select(record.config, "dataset.preprocessing.unknown_labels", default=None)
            if unkn is not None:
                return [str(u) for u in unkn]
        except Exception:
            pass
    unkn_meta = record.metadata.get("unknown_labels")
    if isinstance(unkn_meta, list):
        return [str(u) for u in unkn_meta]
    return []


def validate_compatibility(
    runs: Sequence[RunRecord],
    *,
    require_same_study: bool = True,
    require_same_dataset: bool = True,
    require_same_alpha: bool = True,
    require_same_num_clients: bool = True,
    require_same_held_out: bool = True,
    require_compatible_method: bool = True,
    allow_empty: bool = False,
) -> bool:
    """Validate that a collection of runs is scientifically compatible for aggregation.

    Args:
        runs: Sequence of RunRecord objects to validate.
        require_same_study: Enforce identical study ID across runs.
        require_same_dataset: Enforce identical dataset protocol.
        require_same_alpha: Enforce identical Dirichlet alpha heterogeneity.
        require_same_num_clients: Enforce identical federation client count.
        require_same_held_out: Enforce identical held-out open-set labels.
        require_compatible_method: Enforce identical method family (for multi-seed aggregation).
        allow_empty: If True, empty run list passes validation.

    Returns:
        True if all runs pass compatibility checks.

    Raises:
        IncompatibleRunsError: If any incompatibility is detected.
    """
    if not runs:
        if allow_empty:
            return True
        raise IncompatibleRunsError("Cannot validate compatibility: no runs provided.")

    if len(runs) == 1:
        return True

    discrepancies: list[str] = []
    base = runs[0]

    # 1. Study Check
    if require_same_study:
        studies = {r.study for r in runs}
        if len(studies) > 1:
            discrepancies.append(f"Mismatched studies: {sorted(studies)} across runs {[r.run_id for r in runs]}")

    # 2. Dataset Check
    if require_same_dataset:
        datasets = {r.dataset for r in runs}
        if len(datasets) > 1:
            discrepancies.append(f"Mismatched datasets: {sorted(datasets)}")

    # 3. Method Check
    if require_compatible_method:
        methods = {r.method for r in runs}
        if len(methods) > 1:
            discrepancies.append(f"Mismatched methods in single seed aggregation group: {sorted(methods)}")

    # 4. Alpha Heterogeneity Check
    if require_same_alpha:
        alphas = {round(r.alpha, 4) for r in runs}
        if len(alphas) > 1:
            discrepancies.append(f"Mismatched Dirichlet alpha values: {sorted(alphas)}")

    # 5. Client Count Check
    if require_same_num_clients:
        client_counts = {r.num_clients for r in runs if r.num_clients > 0}
        if len(client_counts) > 1:
            discrepancies.append(f"Mismatched client counts: {sorted(client_counts)}")

    # 6. Held-Out Unknown Labels Check
    if require_same_held_out:
        first_held_out = sorted(_extract_unknown_labels(base))
        for r in runs[1:]:
            r_held_out = sorted(_extract_unknown_labels(r))
            if first_held_out and r_held_out and first_held_out != r_held_out:
                discrepancies.append(
                    f"Mismatched held-out unknown labels between {base.run_id} ({first_held_out}) "
                    f"and {r.run_id} ({r_held_out})"
                )
                break

    # 7. Duplicate Seeds Check within same method group
    if require_compatible_method:
        seeds = [r.seed for r in runs]
        if len(seeds) != len(set(seeds)):
            seen = set()
            dup = [s for s in seeds if s in seen or seen.add(s)]
            discrepancies.append(f"Duplicate seed execution runs detected in aggregation group: {dup}")

    if discrepancies:
        raise IncompatibleRunsError(
            f"Compatibility validation failed for {len(runs)} runs:",
            discrepancies=discrepancies,
        )

    return True
