"""Communication measurement and tensor transmission tracker."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class TransmittedTensorRecord:
    """Record of an individual transmitted tensor."""

    round_num: int
    client_id: str | int
    direction: str  # "uplink" or "downlink"
    tensor_name: str
    shape: list[int]
    dtype: str
    numel: int
    bytes: int


class CommunicationTracker:
    """Live measurement of actual transmitted tensors during federated rounds."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.records: list[TransmittedTensorRecord] = []
        self.round_summaries: list[dict[str, Any]] = []
        self.cumulative_downlink_bytes: int = 0
        self.cumulative_uplink_bytes: int = 0

    def record_tensor(
        self,
        *,
        round_num: int,
        client_id: str | int,
        direction: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> TransmittedTensorRecord:
        """Record a single transmitted tensor."""
        numel = int(tensor.numel())
        element_size = tensor.element_size() if hasattr(tensor, "element_size") else 4
        byte_count = numel * element_size

        record = TransmittedTensorRecord(
            round_num=round_num,
            client_id=str(client_id),
            direction=direction.lower(),
            tensor_name=tensor_name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            numel=numel,
            bytes=byte_count,
        )
        self.records.append(record)

        if record.direction == "uplink":
            self.cumulative_uplink_bytes += byte_count
        else:
            self.cumulative_downlink_bytes += byte_count

        return record

    def record_state_dict(
        self,
        *,
        round_num: int,
        client_id: str | int,
        direction: str,
        state_dict: dict[str, torch.Tensor | Any],
    ) -> int:
        """Record all tensors within a state_dict and return total bytes."""
        total_bytes = 0
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                rec = self.record_tensor(
                    round_num=round_num,
                    client_id=client_id,
                    direction=direction,
                    tensor_name=name,
                    tensor=param,
                )
                total_bytes += rec.bytes
        return total_bytes

    def end_round(self, round_num: int) -> dict[str, int]:
        """Summarize communication for the round."""
        round_records = [r for r in self.records if r.round_num == round_num]
        downlink = sum(r.bytes for r in round_records if r.direction == "downlink")
        uplink = sum(r.bytes for r in round_records if r.direction == "uplink")
        total = downlink + uplink

        summary = {
            "round": round_num,
            "communication/downlink_bytes": downlink,
            "communication/uplink_bytes": uplink,
            "communication/round_bytes": total,
            "communication/cumulative_bytes": self.cumulative_downlink_bytes + self.cumulative_uplink_bytes,
        }
        self.round_summaries.append(summary)

        if self.output_dir is not None:
            self.save(self.output_dir)

        return summary

    def save(self, output_dir: Path) -> None:
        """Persist communication metrics to metrics/communication_round.csv."""
        out = Path(output_dir)
        metrics_dir = out / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if self.round_summaries:
            df_summary = pd.DataFrame(self.round_summaries)
            df_summary.to_csv(metrics_dir / "communication_round.csv", index=False)

        if self.records:
            df_details = pd.DataFrame([asdict(r) for r in self.records])
            df_details.to_csv(metrics_dir / "communication_tensors.csv", index=False)
