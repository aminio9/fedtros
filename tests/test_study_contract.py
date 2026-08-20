from pathlib import Path

from src.infrastructure.study import CANONICAL_SEEDS, expand_study_matrix, load_study_config

ROOT = Path(__file__).resolve().parents[1]


def _study(name):
    return load_study_config(name, ROOT)


def test_every_declared_study_yaml_loads():
    study_dir = ROOT / "src" / "configs" / "study"
    files = sorted(study_dir.glob("*.yaml"))
    assert files
    loaded = [load_study_config(path, ROOT) for path in files]
    assert len(loaded) == len(files)
    assert all(item.get("study_id") for item in loaded)


def test_headline_seed_policy():
    assert tuple(CANONICAL_SEEDS) == (17, 42, 73, 101, 137)
    for study in ("E1-IID-CS", "E2-IID-OSR", "E3-NIID-CS", "E4-NIID-FOSR", "E5-DATASET", "E6-SCALE", "E7-EFFICIENCY", "E8-LOAO"):
        assert list(_study(study)["seeds"]) == list(CANONICAL_SEEDS)


def test_e3_alpha_matrix():
    cfg = _study("E3-NIID-CS")
    assert list(map(float, cfg["alphas"])) == [1.0, 0.5, 0.1]
    assert set(cfg["methods"]) == {"fedtros_pr", "fedavg", "fedprox"}
    assert cfg["known_labels"] == ["Normal", "BP", "DoS", "MitM", "FoT"]


def test_e1_closed_set_uses_all_source_labels():
    cfg = _study("E1-IID-CS")
    assert len(cfg["known_labels_by_dataset"]["bnat"]) == 5
    assert len(cfg["known_labels_by_dataset"]["btat"]) == 7
    plans = expand_study_matrix(cfg, stage="smoke", seeds=[42], project_root=ROOT)
    assert {p.overrides["model.num_classes"] for p in plans if p.dataset == "bnat"} == {5}
    assert {p.overrides["model.num_classes"] for p in plans if p.dataset == "btat"} == {7}


def test_e6_client_scalability_matrix():
    cfg = _study("E6-SCALE")
    assert cfg["num_clients_values"] == [10, 50, 100]
    assert cfg["smoke_num_clients_values"] == [2, 3, 4]
    assert float(cfg["alphas"][0]) == 0.5
    assert int(cfg["base_overrides"]["federated.num_rounds"]) == 100


def test_smoke_client_matrix_is_tiny_without_mutating_paper_matrix():
    e3 = expand_study_matrix(_study("E3-NIID-CS"), stage="smoke", seeds=[42], project_root=ROOT)
    assert {run.num_clients for run in e3} == {2}
    assert all(run.overrides["federated.num_clients"] == 2 for run in e3)
    assert all("partitions/smoke/bnat" in run.partition_file.replace("\\", "/") for run in e3)
    assert all(run.overrides["open_set.calibration.min_samples_per_class"] == 2 for run in e3)

    e6_smoke = expand_study_matrix(_study("E6-SCALE"), stage="smoke", seeds=[42], project_root=ROOT)
    assert {run.num_clients for run in e6_smoke} == {2, 3, 4}

    e6_paper = expand_study_matrix(_study("E6-SCALE"), stage="paper_final", seeds=[42], project_root=ROOT)
    assert {run.num_clients for run in e6_paper} == {10, 50, 100}


def test_e8_holdouts_keep_normal_known():
    cfg = _study("E8-LOAO")
    unknowns = [x[0] for x in cfg["unknown_label_sets"]]
    assert unknowns == ["BP", "DoS", "MitM", "FoT"]
    for unknown in unknowns:
        known = cfg["known_labels_by_unknown"][unknown]
        assert "Normal" in known
        assert unknown not in known


def test_a4_detector_variants():
    cfg = _study("A4-PR")
    variants = [v["name"] for v in cfg["variants"]]
    assert variants == ["msp", "energy", "positive_only", "boundary_raw", "full_rank"]


def test_paired_partition_path_is_method_independent():
    cfg = _study("E3-NIID-CS")
    plans = expand_study_matrix(cfg, stage="paper_final", seeds=[42], project_root=ROOT)
    a01 = [p for p in plans if abs(p.alpha - 0.1) < 1e-9]
    assert len({p.partition_file for p in a01}) == 1
