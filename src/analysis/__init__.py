"""Canonical analysis and results package for FedTROS research.

Provides a unified, data-first scientific interface for query, ingestion,
compatibility validation, statistical aggregation, table generation,
and publication-data export. Publication rendering lives in the separate plots repository.
"""

from __future__ import annotations

from src.analysis.aggregation import (
    AggregatedGroup,
    MetricSummary,
    aggregate_runs,
    compute_metric_stats,
    compute_paired_deltas,
    compute_temporal_last_rounds_stats,
)
from src.analysis.export import (
    export_client_level_contract,
    export_communication_contract,
    export_osr_sample_contract,
    export_runtime_contract,
    build_efficiency_curve,
    generate_provenance_manifest,
)
from src.analysis.loaders import RunRecord, is_run_completed, load_run
from src.analysis.query import query_runs
from src.analysis.statistics import (
    SignificanceReport,
    compare_paired_significance,
    compute_cohens_d,
    format_p_value,
)
from src.analysis.tables import (
    build_ablation_table,
    build_e1_iid_table,
    build_e3_non_iid_table,
    build_e4_open_set_table,
    build_e5_multidataset_table,
    export_all_paper_tables,
)
from src.analysis.validation import IncompatibleRunsError, validate_compatibility

__all__ = [
    # Data Loaders & Query
    "RunRecord",
    "load_run",
    "is_run_completed",
    "query_runs",
    # Validation
    "validate_compatibility",
    "IncompatibleRunsError",
    # Aggregation & Statistics
    "AggregatedGroup",
    "MetricSummary",
    "aggregate_runs",
    "compute_metric_stats",
    "compute_paired_deltas",
    "compute_temporal_last_rounds_stats",
    "SignificanceReport",
    "compare_paired_significance",
    "compute_cohens_d",
    "format_p_value",
    # Paper Tables
    "build_e1_iid_table",
    "build_e3_non_iid_table",
    "build_e4_open_set_table",
    "build_e5_multidataset_table",
    "build_ablation_table",
    "export_all_paper_tables",
    # Data Contracts & Provenance
    "export_osr_sample_contract",
    "export_client_level_contract",
    "export_communication_contract",
    "export_runtime_contract",
    "build_efficiency_curve",
    "generate_provenance_manifest",
]
