"""Unified dataset loading — synthetic by default, optional JSONL for real data."""

from __future__ import annotations

import json
from pathlib import Path

from simulator.config.simulator_config import DatasetConfig
from simulator.data.synthetic import RequestData, SyntheticDataGenerator


class DatasetLoader:
    """Loads request data from synthetic generation or real JSONL files.

    Real dataset format (one JSON object per line)::

        {"prompt_token_ids": [1,2,3], "completion_token_ids": [4,5,6]}
    """

    def __init__(self, config: DatasetConfig, seed: int = 42):
        self._config = config
        self._seed = seed

    def load(self) -> list[RequestData]:
        """Return all request data."""
        if self._config.source == "synthetic":
            generator = SyntheticDataGenerator(
                self._config.synthetic, seed=self._seed
            )
            return generator.generate()
        else:
            return self._load_real_dataset(self._config.real_dataset_path)

    @staticmethod
    def _load_real_dataset(path: str | None) -> list[RequestData]:
        """Load from a JSONL file."""
        if path is None:
            raise ValueError("real_dataset_path must be set when source='real'")

        results: list[RequestData] = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                results.append(
                    RequestData(
                        request_id=obj.get("request_id", f"req-{i:06d}"),
                        prompt_token_ids=obj["prompt_token_ids"],
                        ground_truth_output=obj.get(
                            "completion_token_ids",
                            obj.get("ground_truth_output", []),
                        ),
                        arrival_time=obj.get("arrival_time", 0.0),
                    )
                )
        return results
