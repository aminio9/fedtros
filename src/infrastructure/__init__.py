"""Core infrastructure package for FedTROS-PR experiment running and tracking."""

from src.infrastructure.checkpointing import (
    CheckpointState,
    IncompatibleCheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from src.infrastructure.hardware import (
    get_cpu_and_ram_info,
    get_full_environment_provenance,
    get_gpu_info,
    get_installed_packages,
)
from src.infrastructure.instrumentation import (
    CommunicationTracker,
    RuntimeTracker,
    TransmittedTensorRecord,
)
from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.manifests import (
    RunManifest,
    RunStatus,
    create_initial_run_manifest,
    initialize_run_directories,
    update_run_manifest_status,
    write_canonical_manifests,
)
from src.infrastructure.run_id import (
    RunCollisionError,
    compute_scientific_config_hash,
    generate_run_id,
    validate_run_collision,
)
from src.infrastructure.study import (
    CANONICAL_SEEDS,
    STAGE_PROFILES,
    DryRunSummary,
    PlannedRun,
    expand_study_matrix,
    filter_missing_runs,
    get_paired_partition_path,
    load_study_config,
    perform_dry_run,
)
from src.infrastructure.tracking import (
    ExperimentTracker,
    NullTracker,
    create_tracker,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    # Hardware
    "get_gpu_info",
    "get_cpu_and_ram_info",
    "get_installed_packages",
    "get_full_environment_provenance",
    # Run ID & Hashing
    "compute_scientific_config_hash",
    "generate_run_id",
    "validate_run_collision",
    "RunCollisionError",
    # Manifests
    "RunStatus",
    "RunManifest",
    "create_initial_run_manifest",
    "update_run_manifest_status",
    "initialize_run_directories",
    "write_canonical_manifests",
    # Tracking
    "ExperimentTracker",
    "NullTracker",
    "create_tracker",
    # Checkpointing
    "CheckpointState",
    "IncompatibleCheckpointError",
    "save_checkpoint",
    "load_checkpoint",
    # Instrumentation
    "CommunicationTracker",
    "TransmittedTensorRecord",
    "RuntimeTracker",
    # Study
    "CANONICAL_SEEDS",
    "STAGE_PROFILES",
    "PlannedRun",
    "DryRunSummary",
    "load_study_config",
    "expand_study_matrix",
    "perform_dry_run",
    "filter_missing_runs",
    "get_paired_partition_path",
]
