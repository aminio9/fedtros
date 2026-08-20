"""Statistical hypothesis testing and effect size calculations for Q1 paper defense.

Provides paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes,
and confidence interval comparisons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from src.analysis.loaders import RunRecord

logger = logging.getLogger(__name__)


@dataclass
class SignificanceReport:
    """Statistical significance test report between candidate and baseline."""

    metric: str
    n_pairs: int
    candidate_mean: float
    baseline_mean: float
    delta_mean: float
    delta_ci95_low: float
    delta_ci95_high: float
    t_statistic: float
    p_value_t_test: float
    wilcoxon_stat: float | None
    p_value_wilcoxon: float | None
    cohens_d: float
    is_statistically_significant_05: bool
    is_statistically_significant_01: bool

    def format_latex_row(self, metric_display: str | None = None) -> str:
        """Format as LaTeX table row with p-value and effect size."""
        m_name = metric_display or self.metric
        p_str = format_p_value(self.p_value_t_test)
        d_sign = "+" if self.delta_mean >= 0 else ""
        return (
            f"{m_name} & {self.candidate_mean:.2f} & {self.baseline_mean:.2f} & "
            f"{d_sign}{self.delta_mean:.2f} [{self.delta_ci95_low:.2f}, {self.delta_ci95_high:.2f}] & "
            f"{p_str} & {self.cohens_d:.2f} \\\\"
        )


def format_p_value(p: float) -> str:
    """Format p-value with standard academic formatting."""
    if math.isnan(p):
        return "N/A"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def compute_cohens_d(x: Sequence[float], y: Sequence[float], paired: bool = True) -> float:
    """Compute Cohen's d effect size."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return 0.0

    if paired:
        diff = x_arr - y_arr
        s_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
        return float(np.mean(diff) / s_diff) if s_diff > 1e-12 else 0.0
    else:
        nx, ny = len(x_arr), len(y_arr)
        vx, vy = np.var(x_arr, ddof=1), np.var(y_arr, ddof=1)
        s_pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
        return float((np.mean(x_arr) - np.mean(y_arr)) / s_pooled) if s_pooled > 1e-12 else 0.0


def compare_paired_significance(
    candidate_runs: Sequence[RunRecord],
    baseline_runs: Sequence[RunRecord],
    metric_key: str,
) -> SignificanceReport | None:
    """Perform comprehensive paired significance testing between candidate and baseline runs."""
    cand_by_seed = {r.seed: r for r in candidate_runs}
    base_by_seed = {r.seed: r for r in baseline_runs}
    common_seeds = sorted(set(cand_by_seed.keys()) & set(base_by_seed.keys()))

    if len(common_seeds) < 2:
        logger.warning("Need at least 2 common seeds for paired statistical significance test.")
        return None

    c_vals = []
    b_vals = []
    diffs = []
    for s in common_seeds:
        c = cand_by_seed[s].get_metric([metric_key])
        b = base_by_seed[s].get_metric([metric_key])
        if c is not None and b is not None:
            c_vals.append(c)
            b_vals.append(b)
            diffs.append(c - b)

    n = len(diffs)
    if n < 2:
        return None

    c_arr = np.asarray(c_vals, dtype=float)
    b_arr = np.asarray(b_vals, dtype=float)
    d_arr = np.asarray(diffs, dtype=float)

    delta_mean = float(np.mean(d_arr))
    delta_std = float(np.std(d_arr, ddof=1))
    sem = delta_std / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_low = delta_mean - t_crit * sem
    ci_high = delta_mean + t_crit * sem

    # Paired t-test
    t_res = stats.ttest_rel(c_arr, b_arr)
    t_stat = float(t_res.statistic) if math.isfinite(t_res.statistic) else 0.0
    p_t = float(t_res.pvalue) if math.isfinite(t_res.pvalue) else 1.0

    # Wilcoxon signed rank test
    w_stat, p_w = None, None
    if n >= 5 and not np.all(d_arr == 0):
        try:
            w_res = stats.wilcoxon(c_arr, b_arr)
            w_stat = float(w_res.statistic)
            p_w = float(w_res.pvalue)
        except Exception:
            pass

    d_effect = compute_cohens_d(c_arr, b_arr, paired=True)

    return SignificanceReport(
        metric=metric_key,
        n_pairs=n,
        candidate_mean=float(np.mean(c_arr)),
        baseline_mean=float(np.mean(b_arr)),
        delta_mean=delta_mean,
        delta_ci95_low=ci_low,
        delta_ci95_high=ci_high,
        t_statistic=t_stat,
        p_value_t_test=p_t,
        wilcoxon_stat=w_stat,
        p_value_wilcoxon=p_w,
        cohens_d=d_effect,
        is_statistically_significant_05=p_t < 0.05,
        is_statistically_significant_01=p_t < 0.01,
    )
