from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BTAT_REPOSITORY = "https://github.com/avitech-vnu/BTAT.git"
BTAT_COMMIT = "3501f2514a55a1cce080ae55b1be66f1295281a6"
TONIOT_OFFICIAL_PAGE = "https://research.unsw.edu.au/projects/toniot-datasets"
TONIOT_MIRROR_COMMIT = "ed5bc29125a5d057be27b8b251ec3170adaa9cf4"
TONIOT_MIRROR_URL = (
    "https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/"
    f"{TONIOT_MIRROR_COMMIT}/train_test_network.csv?download=true"
)
TONIOT_SHA256 = "26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974"
CICIDS_OFFICIAL_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
CICIDS_ARCHIVE_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/"
    "CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
)

BTAT_LABELS = ("Normal", "DoS", "OaU", "FoT", "Re", "DeC", "FDV")
TONIOT_LABELS = (
    "normal",
    "backdoor",
    "ddos",
    "dos",
    "injection",
    "mitm",
    "password",
    "ransomware",
    "scanning",
    "xss",
)
CICIDS_LABELS = (
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack-Brute Force",
    "Web Attack-Sql Injection",
    "Web Attack-XSS",
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "cf-marlos-dataset-preparer/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        append = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    return destination


def _parse_integer(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return 0


def _hex_bytes(value: Any) -> bytes:
    text = str(value or "").strip()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) % 2:
        text = "0" + text
    try:
        return bytes.fromhex(text)
    except ValueError:
        return b""


def evm_opcode_features(input_value: Any) -> dict[str, float | int]:
    bytecode = _hex_bytes(input_value)
    counts = np.zeros(256, dtype=np.int64)
    position = 0
    opcode_count = 0
    while position < len(bytecode):
        opcode = bytecode[position]
        counts[opcode] += 1
        opcode_count += 1
        position += 1
        if 0x60 <= opcode <= 0x7F:
            position += opcode - 0x5F

    probabilities = counts[counts > 0] / max(opcode_count, 1)
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else 0.0
    features: dict[str, float | int] = {
        "bytecode_length": len(bytecode),
        "opcode_count": opcode_count,
        "opcode_entropy": entropy,
        "nonzero_byte_ratio": (
            float(np.count_nonzero(np.frombuffer(bytecode, dtype=np.uint8)) / len(bytecode))
            if bytecode
            else 0.0
        ),
    }
    for opcode, count in enumerate(counts.tolist()):
        features[f"opcode_{opcode:02x}"] = float(count / max(opcode_count, 1))
    return features


def btat_record_to_row(record: dict[str, Any], label: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gas": _parse_integer(record.get("gas")),
        "gas_price": _parse_integer(record.get("gasPrice")),
        "value": _parse_integer(record.get("value")),
    }
    row.update(evm_opcode_features(record.get("input")))
    row["label"] = label
    return row


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
            return
    values = payload if isinstance(payload, list) else [payload]
    for value in values:
        if isinstance(value, dict):
            yield value


def _convert_btat_file(item: tuple[Path, str]) -> list[dict[str, Any]]:
    path, label = item
    return [btat_record_to_row(record, label) for record in _iter_json_records(path)]


def _truncate_incomplete_csv_row(path: Path) -> None:
    """Remove only a torn final row left by an interrupted writer."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        position = handle.tell() - 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) == b"\n":
                handle.truncate(position + 1)
                return
            position -= 1
        handle.truncate(0)


def _prepare_btat_checkout(repository_dir: Path) -> None:
    repository_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (repository_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", BTAT_REPOSITORY, str(repository_dir)],
            check=True,
        )
    safe_directory = repository_dir.resolve().as_posix()
    git = ["git", "-c", f"safe.directory={safe_directory}", "-C", str(repository_dir)]
    current_commit = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != BTAT_COMMIT:
        subprocess.run([*git, "fetch", "origin", BTAT_COMMIT], check=True)
    subprocess.run(
        [*git, "checkout", "--detach", BTAT_COMMIT],
        check=True,
    )
    resolved_commit = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_commit != BTAT_COMMIT:
        raise RuntimeError(
            f"BTAT checkout mismatch: expected={BTAT_COMMIT}, actual={resolved_commit}"
        )


def convert_btat(repository_dir: Path, output_csv: Path) -> dict[str, Any]:
    aliases = {
        "Normal": "Normal",
        "DoS": "DoS",
        "OaU": "OaU",
        "FoT": "FoT",
        "Reentrancy": "Re",
        "Re": "Re",
        "Delegatecall": "DeC",
        "DeC": "DeC",
        "FDV": "FDV",
    }
    opcode_columns = [f"opcode_{value:02x}" for value in range(256)]
    columns = [
        "gas",
        "gas_price",
        "value",
        "bytecode_length",
        "opcode_count",
        "opcode_entropy",
        "nonzero_byte_ratio",
        *opcode_columns,
        "label",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = output_csv.with_suffix(output_csv.suffix + ".part")
    counts: Counter[str] = Counter()
    if partial_csv.exists():
        _truncate_incomplete_csv_row(partial_csv)
        with partial_csv.open("r", newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != columns:
            raise ValueError(
                f"BTAT partial schema mismatch in {partial_csv}; remove it and rerun preparation."
            )
        existing_labels = pd.read_csv(partial_csv, usecols=["label"], low_memory=False)["label"]
        counts.update(str(value) for value in existing_labels)

    write_header = not partial_csv.exists() or partial_csv.stat().st_size == 0
    with partial_csv.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        if write_header:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="btat-reader") as executor:
            for source_name, label in aliases.items():
                source_dir = repository_dir / source_name
                if not source_dir.is_dir():
                    continue
                paths = [
                    path
                    for path in sorted(source_dir.rglob("*"))
                    if path.is_file()
                    and not path.name.startswith(".")
                    and path.suffix.lower() != ".md"
                ]
                completed = int(counts[label])
                if completed > len(paths):
                    raise ValueError(
                        f"BTAT partial row count for {label} exceeds source files: "
                        f"partial={completed}, source={len(paths)}"
                    )
                paths = paths[completed:]
                for offset in range(0, len(paths), 4096):
                    batch = ((path, label) for path in paths[offset : offset + 4096])
                    for rows in executor.map(_convert_btat_file, batch):
                        writer.writerows(rows)
                        counts[label] += len(rows)
    _validate_labels(counts, BTAT_LABELS, dataset="BTAT")
    partial_csv.replace(output_csv)
    return {"rows": int(sum(counts.values())), "class_counts": dict(sorted(counts.items()))}


TONIOT_DROP_COLUMNS = {
    "ts",
    "src_ip",
    "dst_ip",
    "dns_query",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_addl",
    "label",
}


def convert_toniot(source_csv: Path, output_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(source_csv, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "type" not in frame.columns:
        raise KeyError("ToN-IoT source must contain the multiclass 'type' column.")
    frame["type"] = frame["type"].astype(str).str.strip().str.lower()
    frame = frame.drop(columns=[column for column in TONIOT_DROP_COLUMNS if column in frame])
    frame = frame.replace([np.inf, -np.inf], np.nan).drop_duplicates(keep="first")
    counts = Counter(str(value) for value in frame["type"])
    _validate_labels(counts, TONIOT_LABELS, dataset="ToN-IoT")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = output_csv.with_suffix(output_csv.suffix + ".part")
    partial_csv.unlink(missing_ok=True)
    frame.to_csv(partial_csv, index=False, encoding="utf-8")
    partial_csv.replace(output_csv)
    return {"rows": len(frame), "class_counts": dict(sorted(counts.items()))}


def normalize_cicids_label(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    lowered = text.lower()
    if lowered.startswith("web attack"):
        if "brute force" in lowered:
            return "Web Attack-Brute Force"
        if "sql injection" in lowered:
            return "Web Attack-Sql Injection"
        if "xss" in lowered:
            return "Web Attack-XSS"
    lookup = {label.lower(): label for label in CICIDS_LABELS}
    return lookup.get(lowered, text)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        handle.extractall(destination)


def convert_cicids2017(csv_files: Iterable[Path], output_csv: Path) -> dict[str, Any]:
    paths = sorted(Path(path) for path in csv_files)
    if len(paths) != 8:
        raise ValueError(f"Expected exactly 8 CIC-IDS2017 CSV files, found {len(paths)}.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = output_csv.with_suffix(output_csv.suffix + ".part")
    partial_csv.unlink(missing_ok=True)
    wrote_header = False
    counts: Counter[str] = Counter()
    seen_hashes: set[int] = set()
    total_rows = 0
    canonical_columns: list[str] | None = None
    for path in paths:
        for frame in pd.read_csv(path, chunksize=100_000, low_memory=False):
            frame.columns = [str(column).strip() for column in frame.columns]
            label_candidates = [column for column in frame if column.lower() == "label"]
            if len(label_candidates) != 1:
                raise KeyError(f"Expected one Label column in {path}, found {label_candidates}.")
            frame = frame.rename(columns={label_candidates[0]: "Label"})
            frame["Label"] = frame["Label"].map(normalize_cicids_label)
            frame = frame.drop(
                columns=[
                    column
                    for column in ("Flow ID", "Source IP", "Destination IP", "Timestamp")
                    if column in frame
                ]
            )
            frame = frame.replace([np.inf, -np.inf], np.nan)
            if canonical_columns is None:
                canonical_columns = list(frame.columns)
            elif list(frame.columns) != canonical_columns:
                raise ValueError(f"CIC-IDS2017 schema mismatch in {path}.")
            hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
            keep = np.fromiter((int(value) not in seen_hashes for value in hashes), dtype=bool)
            frame = frame.loc[keep]
            kept_hashes = hashes[keep]
            seen_hashes.update(int(value) for value in kept_hashes)
            if frame.empty:
                continue
            counts.update(str(value) for value in frame["Label"])
            frame.to_csv(
                partial_csv,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                index=False,
                encoding="utf-8",
            )
            wrote_header = True
            total_rows += len(frame)
    _validate_labels(counts, CICIDS_LABELS, dataset="CIC-IDS2017")
    partial_csv.replace(output_csv)
    return {"rows": total_rows, "class_counts": dict(sorted(counts.items()))}


def _validate_labels(counts: Counter[str], expected: Iterable[str], *, dataset: str) -> None:
    actual = set(counts)
    expected_set = set(expected)
    if actual != expected_set:
        raise ValueError(
            f"{dataset} label validation failed: missing={sorted(expected_set - actual)}, "
            f"unexpected={sorted(actual - expected_set)}"
        )
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"{dataset} contains an empty class.")


def _write_manifest(
    *,
    dataset: str,
    canonical_csv: Path,
    source: dict[str, Any],
    conversion: dict[str, Any],
) -> Path:
    manifest = {
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "canonical_csv": str(canonical_csv),
        "canonical_sha256": sha256_file(canonical_csv),
        **conversion,
    }
    manifest_path = canonical_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _canonical_is_prepared(
    canonical_csv: Path,
    *,
    dataset: str,
    expected_labels: Iterable[str],
) -> bool:
    manifest_path = canonical_csv.with_suffix(".manifest.json")
    if not canonical_csv.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("dataset") != dataset or int(manifest.get("rows", 0)) <= 0:
        return False
    if set(manifest.get("class_counts", {})) != set(expected_labels):
        return False
    expected_digest = str(manifest.get("canonical_sha256", ""))
    return bool(expected_digest) and sha256_file(canonical_csv) == expected_digest


def prepare_btat(raw_root: Path) -> Path:
    source_dir = raw_root / "source" / "btat"
    repository_dir = source_dir / "repository"
    _prepare_btat_checkout(repository_dir)
    canonical = raw_root / "BTAT.csv"
    if _canonical_is_prepared(canonical, dataset="BTAT", expected_labels=BTAT_LABELS):
        return canonical
    conversion = convert_btat(repository_dir, canonical)
    _write_manifest(
        dataset="BTAT",
        canonical_csv=canonical,
        source={"repository": BTAT_REPOSITORY, "commit": BTAT_COMMIT},
        conversion=conversion,
    )
    return canonical


def prepare_toniot(raw_root: Path, *, source_url: str = TONIOT_MIRROR_URL) -> Path:
    source_dir = raw_root / "source" / "toniot"
    source_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        path
        for path in source_dir.glob("*.csv")
        if path.name.lower() == "train_test_network.csv"
    ]
    source_csv = candidates[0] if candidates else source_dir / "train_test_network.csv"
    digest = sha256_file(source_csv) if source_csv.exists() else ""
    if digest != TONIOT_SHA256:
        if source_csv.exists():
            raise ValueError(
                f"ToN-IoT checksum mismatch: expected={TONIOT_SHA256}, actual={digest}, "
                f"source={source_csv}"
            )
        download_resumable(source_url, source_csv)
        digest = sha256_file(source_csv)
    if digest != TONIOT_SHA256:
        raise ValueError(f"ToN-IoT checksum mismatch: expected={TONIOT_SHA256}, actual={digest}")
    canonical = raw_root / "ToN-IoT.csv"
    if _canonical_is_prepared(canonical, dataset="ToN-IoT", expected_labels=TONIOT_LABELS):
        return canonical
    conversion = convert_toniot(source_csv, canonical)
    _write_manifest(
        dataset="ToN-IoT",
        canonical_csv=canonical,
        source={
            "official_page": TONIOT_OFFICIAL_PAGE,
            "verified_mirror": source_url,
            "mirror_commit": TONIOT_MIRROR_COMMIT,
            "source_sha256": digest,
        },
        conversion=conversion,
    )
    return canonical


def prepare_cicids2017(raw_root: Path, *, source_url: str = CICIDS_ARCHIVE_URL) -> Path:
    source_dir = raw_root / "source" / "cicids2017"
    archive = source_dir / "MachineLearningCSV.zip"
    csv_files = sorted(source_dir.rglob("*.csv"))
    if len(csv_files) != 8:
        if not archive.exists() or not zipfile.is_zipfile(archive):
            archive.unlink(missing_ok=True)
            download_resumable(source_url, archive)
        if not zipfile.is_zipfile(archive):
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                "The CIC endpoint did not return MachineLearningCSV.zip. The current official "
                f"download form is {CICIDS_OFFICIAL_PAGE}; download and extract the archive "
                f"below {source_dir}, then rerun this command."
            )
        extracted = source_dir / "extracted"
        _safe_extract_zip(archive, extracted)
        csv_files = sorted(extracted.rglob("*.csv"))
    canonical = raw_root / "CIC-IDS2017.csv"
    if _canonical_is_prepared(
        canonical,
        dataset="CIC-IDS2017",
        expected_labels=CICIDS_LABELS,
    ):
        return canonical
    conversion = convert_cicids2017(csv_files, canonical)
    source_files = [
        {
            "path": str(path.relative_to(source_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in csv_files
    ]
    _write_manifest(
        dataset="CIC-IDS2017",
        canonical_csv=canonical,
        source={
            "official_page": CICIDS_OFFICIAL_PAGE,
            "archive_url": source_url if archive.exists() else None,
            "archive_sha256": sha256_file(archive) if archive.exists() else None,
            "csv_files": source_files,
        },
        conversion=conversion,
    )
    return canonical


def prepare_external_dataset(dataset: str, *, raw_root: Path) -> list[Path]:
    selected = dataset.lower()
    if selected not in {"btat", "toniot", "cicids2017", "all"}:
        raise ValueError(f"Unsupported external dataset: {dataset}")
    outputs: list[Path] = []
    if selected in {"btat", "all"}:
        outputs.append(prepare_btat(raw_root))
    if selected in {"toniot", "all"}:
        outputs.append(prepare_toniot(raw_root))
    if selected in {"cicids2017", "all"}:
        outputs.append(prepare_cicids2017(raw_root))
    return outputs
