"""Multi-seed statistical aggregation and paired delta calculations (Item C8).

Supports calculating mean, across-seed standard deviation, 95% confidence intervals
using Student's t-distribution, paired baseline deltas, and completion metrics.
Strictly separates across-seed variance from round-to-round temporal variance.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from scipy import stats

from src.analysis.loaders import RunRecord
from src.analysis.validation import validate_compatibility

logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    """Statistical summary for a single metric aggregated across runs."""

    name: str
    n: int
    mean: float
    std_across_seeds: float
    ci95_margin: float
    ci95_low: float
    ci95_high: float
    min_value: float
    max_value: float
    raw_values: list[float] = field(default_factory=list)

    # Optional paired delta against baseline
    paired_delta_mean: float | None = None
    paired_delta_std: float | None = None
    paired_delta_ci95: float | None = None

    def format_mean_std(self, percent: bool = False, precision: int = 2) -> str:
        """Format as 'mean ± std'."""
        scale = 100.0 if percent else 1.0
        m = self.mean * scale
        s = self.std_across_seeds * scale
        return f"{m:.{precision}f} ± {s:.{precision}f}"

    def format_mean_ci(self, percent: bool = False, precision: int = 2) -> str:
        """Format as 'mean ± 95% CI'."""
        scale = 100.0 if percent else 1.0
        m = self.mean * scale
        c = self.ci95_margin * scale
        return f"{m:.{precision}f} ± {c:.{precision}f}"


@dataclass
class AggregatedGroup:
    """Group of runs aggregated across seeds for a single condition."""

    study: str
    stage: str
    method: str
    dataset: str
    alpha: float
    num_clients: int
    completed_count: int
    total_seeds: int
    seeds: list[int]
    run_ids: list[str]
    metrics: dict[str, MetricSummary] = field(default_factory=dict)

    # Temporal variance across last rounds (Strictly separated from across-seed SD)
    temporal_metrics: dict[str, float] = field(default_factory=dict)


def compute_metric_stats(values: Sequence[float], name: str = "metric") -> MetricSummary:
    """Compute mean, across-seed SD, and 95% Student's t CI for an array of values."""
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return MetricSummary(
            name=name,
            n=0,
            mean=np.nan,
            std_across_seeds=np.nan,
            ci95_margin=np.nan,
            ci95_low=np.nan,
            ci95_high=np.nan,
            min_value=np.nan,
            max_value=np.nan,
            raw_values=[],
        )

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1:
        # Student's t-distribution for small sample sizes (degrees of freedom = n - 1)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        sem = std / math.sqrt(n)
        ci_margin = float(t_crit * sem)
    else:
        ci_margin = 0.0

    return MetricSummary(
        name=name,
        n=n,
        mean=mean,
        std_across_seeds=std,
        ci95_margin=ci_margin,
        ci95_low=mean - ci_margin,
        ci95_high=mean + ci_margin,
        min_value=float(np.min(arr)),
        max_value=float(np.max(arr)),
        raw_values=arr.tolist(),
    )


def compute_temporal_last_rounds_stats(
    runs: Sequence[RunRecord],
    metric_name: str,
    last_n_rounds: int = 10,
) -> dict[str, float]:
    """Compute temporal variation over the final N communication rounds across runs.

    NOTE: This is strictly separated from across-seed variance.
    """
    all_tail_values: list[float] = []
    for r in runs:
        hist = r.history
        if hist.empty:
            continue
        # Filter for metric
        if "metric_name" in hist.columns and "metric_value" in hist.columns:
            m_rows = hist[hist["metric_name"] == metric_name]
            if not m_rows.empty:
                vals = (
                    pd.to_numeric(m_rows["metric_value"], errors="coerce")
                    .dropna()
                    .tail(last_n_rounds)
                )
                all_tail_values.extend(vals.tolist())
        elif metric_name in hist.columns:
            vals = pd.to_numeric(hist[metric_name], errors="coerce").dropna().tail(last_n_rounds)
            all_tail_values.extend(vals.tolist())

    if not all_tail_values:
        return {
            f"temporal_last{last_n_rounds}_mean": np.nan,
            f"temporal_last{last_n_rounds}_sd": np.nan,
        }

    return {
        f"temporal_last{last_n_rounds}_mean": float(np.mean(all_tail_values)),
        f"temporal_last{last_n_rounds}_sd": float(
            np.std(all_tail_values, ddof=1) if len(all_tail_values) > 1 else 0.0
        ),
    }


def aggregate_runs(
    runs: Sequence[RunRecord],
    metric_keys: Sequence[str] | None = None,
    validate: bool = True,
) -> AggregatedGroup:
    """Aggregate a collection of compatible seed runs into an AggregatedGroup.

    Args:
        runs: Sequence of RunRecord objects sharing condition (study, method, dataset, alpha).
        metric_keys: Specific metrics to aggregate, or None to aggregate all numeric metrics found.
        validate: If True, execute strict compatibility validation before aggregation.

    Returns:
        AggregatedGroup containing multi-seed statistics.
    """
    if not runs:
        raise ValueError("Cannot aggregate empty list of runs.")

    if validate:
        validate_compatibility(runs, require_compatible_method=True)

    base = runs[0]
    seeds = sorted([r.seed for r in runs])
    run_ids = [r.run_id for r in runs]

    # Collect all metric keys across runs if not explicitly provided
    if metric_keys is None:
        found_keys: set[str] = set()
        for r in runs:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    found_keys.add(k)
        keys_to_aggregate = sorted(found_keys)
    else:
        keys_to_aggregate = list(metric_keys)

    summaries: dict[str, MetricSummary] = {}
    for k in keys_to_aggregate:
        vals = [r.get_metric([k]) for r in runs if r.get_metric([k]) is not None]
        summaries[k] = compute_metric_stats(vals, name=k)

    # Compute separate temporal last-round statistics for key convergence metrics
    temporal = {}
    for tm in ("local_student_accuracy", "round_openset_f1_macro", "round_openset_auroc"):
        temporal.update(compute_temporal_last_rounds_stats(runs, tm, last_n_rounds=10))

    return AggregatedGroup(
        study=base.study,
        stage=base.stage,
        method=base.method,
        dataset=base.dataset,
        alpha=base.alpha,
        num_clients=base.num_clients,
        completed_count=len(runs),
        total_seeds=len(seeds),
        seeds=seeds,
        run_ids=run_ids,
        metrics=summaries,
        temporal_metrics=temporal,
    )


def compute_paired_deltas(
    candidate_runs: Sequence[RunRecord],
    baseline_runs: Sequence[RunRecord],
    metric_keys: Sequence[str],
) -> dict[str, MetricSummary]:
    """Compute paired per-seed differences: Delta = Candidate - Baseline.

    Matches runs on exact seed, computes difference for each seed pair,
    and returns MetricSummary representing the paired delta distribution.
    """
    cand_by_seed = {r.seed: r for r in candidate_runs}
    base_by_seed = {r.seed: r for r in baseline_runs}
    common_seeds = sorted(set(cand_by_seed.keys()) & set(base_by_seed.keys()))

    if not common_seeds:
        logger.warning("No common seeds found for paired delta calculation.")
        return {}

    deltas: dict[str, MetricSummary] = {}
    for m in metric_keys:
        diffs = []
        for s in common_seeds:
            c_val = cand_by_seed[s].get_metric([m])
            b_val = base_by_seed[s].get_metric([m])
            if c_val is not None and b_val is not None:
                diffs.append(c_val - b_val)
        deltas[m] = compute_metric_stats(diffs, name=f"delta_{m}")

    return deltas
