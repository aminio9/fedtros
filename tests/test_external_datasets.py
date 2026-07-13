import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from src.data import external_datasets
from src.data.external_datasets import (
    BTAT_LABELS,
    CICIDS_LABELS,
    TONIOT_LABELS,
    convert_btat,
    convert_cicids2017,
    convert_toniot,
    download_resumable,
    evm_opcode_features,
    prepare_cicids2017,
    prepare_toniot,
)


class _Response(BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_evm_opcode_features_skip_push_payload():
    features = evm_opcode_features("0x6001600201")

    assert features["bytecode_length"] == 5
    assert features["opcode_count"] == 3
    assert features["opcode_60"] == 2 / 3
    assert features["opcode_01"] == 1 / 3
    assert features["opcode_02"] == 0.0


def test_download_resumable_appends_to_interrupted_part(monkeypatch, tmp_path):
    destination = tmp_path / "source.csv"
    destination.with_suffix(".csv.part").write_bytes(b"abc")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response(b"def", status=206)

    monkeypatch.setattr(external_datasets.urllib.request, "urlopen", fake_urlopen)

    download_resumable("https://example.test/source.csv", destination)

    assert destination.read_bytes() == b"abcdef"
    assert requests[0][0].get_header("Range") == "bytes=3-"


def test_prepare_toniot_rejects_checksum_mismatch(monkeypatch, tmp_path):
    def fake_download(_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"not-the-verified-dataset")
        return destination

    monkeypatch.setattr(external_datasets, "download_resumable", fake_download)

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_toniot(tmp_path)


def test_prepare_cicids_rejects_html_download_and_removes_it(monkeypatch, tmp_path):
    def fake_download(_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("<html>download form</html>", encoding="utf-8")
        return destination

    monkeypatch.setattr(external_datasets, "download_resumable", fake_download)

    with pytest.raises(RuntimeError, match="official download form"):
        prepare_cicids2017(tmp_path)

    assert not (tmp_path / "source" / "cicids2017" / "MachineLearningCSV.zip").exists()


def test_convert_btat_creates_seven_class_numeric_csv(tmp_path):
    aliases = {
        "Normal": "Normal",
        "DoS": "DoS",
        "OaU": "OaU",
        "FoT": "FoT",
        "Reentrancy": "Re",
        "Delegatecall": "DeC",
        "FDV": "FDV",
    }
    for source_label in aliases:
        folder = tmp_path / "repository" / source_label
        folder.mkdir(parents=True)
        (folder / "record.json").write_text(
            json.dumps(
                {
                    "gas": "0x5208",
                    "gasPrice": "0x3b9aca00",
                    "value": "0xde0b6b3a7640000",
                    "input": "0x6001600201",
                    "hash": "must-not-be-exported",
                    "from": "must-not-be-exported",
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "BTAT.csv"
    result = convert_btat(tmp_path / "repository", output)
    frame = pd.read_csv(output)

    assert result["rows"] == 7
    assert set(frame["label"]) == set(BTAT_LABELS)
    assert "hash" not in frame
    assert "from" not in frame
    frequency_columns = [
        column for column in frame if column.startswith("opcode_") and len(column) == 9
    ]
    assert len(frequency_columns) == 256


def test_failed_btat_validation_never_publishes_partial_csv(tmp_path):
    folder = tmp_path / "repository" / "Normal"
    folder.mkdir(parents=True)
    (folder / "record.json").write_text('{"input": "0x00"}', encoding="utf-8")
    output = tmp_path / "BTAT.csv"

    with pytest.raises(ValueError, match="label validation failed"):
        convert_btat(tmp_path / "repository", output)

    assert not output.exists()


def test_convert_btat_resumes_an_interrupted_partial_without_duplicates(tmp_path):
    aliases = {
        "Normal": "Normal",
        "DoS": "DoS",
        "OaU": "OaU",
        "FoT": "FoT",
        "Reentrancy": "Re",
        "Delegatecall": "DeC",
        "FDV": "FDV",
    }
    normal = tmp_path / "repository" / "Normal"
    normal.mkdir(parents=True)
    (normal / "record.json").write_text('{"input": "0x00"}', encoding="utf-8")
    output = tmp_path / "BTAT.csv"
    with pytest.raises(ValueError, match="label validation failed"):
        convert_btat(tmp_path / "repository", output)

    for folder_name in set(aliases) - {"Normal"}:
        folder = tmp_path / "repository" / folder_name
        folder.mkdir(parents=True)
        (folder / "record.json").write_text('{"input": "0x00"}', encoding="utf-8")

    result = convert_btat(tmp_path / "repository", output)
    frame = pd.read_csv(output)

    assert result["rows"] == 7
    assert frame["label"].value_counts().to_dict() == {label: 1 for label in aliases.values()}


def test_convert_toniot_drops_identifiers_and_binary_label(tmp_path):
    rows = []
    for index, label in enumerate(TONIOT_LABELS):
        rows.append(
            {
                "src_ip": f"10.0.0.{index}",
                "dst_ip": "10.0.1.1",
                "src_port": 1000 + index,
                "proto": "tcp",
                "service": "http",
                "duration": index + 0.5,
                "dns_query": "sensitive.example",
                "label": int(label != "normal"),
                "type": label.upper(),
            }
        )
    source = tmp_path / "train_test_network.csv"
    output = tmp_path / "ToN-IoT.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    result = convert_toniot(source, output)
    frame = pd.read_csv(output)

    assert result["rows"] == 10
    assert set(frame["type"]) == set(TONIOT_LABELS)
    assert "src_ip" not in frame
    assert "dst_ip" not in frame
    assert "dns_query" not in frame
    assert "label" not in frame


def test_convert_cicids_merges_eight_files_and_normalizes_labels(tmp_path):
    rows = [
        {"Destination Port": index, "Flow Duration": index + 1, "Label": label}
        for index, label in enumerate(CICIDS_LABELS)
    ]
    rows[-3]["Label"] = "Web Attack � Brute Force"
    rows[-2]["Label"] = "Web Attack � Sql Injection"
    rows[-1]["Label"] = "Web Attack � XSS"
    files: list[Path] = []
    for file_index in range(8):
        path = tmp_path / f"part_{file_index}.csv"
        part = rows[file_index::8]
        pd.DataFrame(part).to_csv(path, index=False)
        files.append(path)

    output = tmp_path / "CIC-IDS2017.csv"
    result = convert_cicids2017(files, output)
    frame = pd.read_csv(output)

    assert result["rows"] == 15
    assert set(frame["Label"]) == set(CICIDS_LABELS)


def test_convert_cicids_rejects_schema_drift(tmp_path):
    files = []
    for index in range(8):
        frame = pd.DataFrame({"feature": [index], "Label": [CICIDS_LABELS[index]]})
        if index == 4:
            frame["unexpected"] = 1
        path = tmp_path / f"part_{index}.csv"
        frame.to_csv(path, index=False)
        files.append(path)

    with pytest.raises(ValueError, match="schema mismatch"):
        convert_cicids2017(files, tmp_path / "CIC-IDS2017.csv")
