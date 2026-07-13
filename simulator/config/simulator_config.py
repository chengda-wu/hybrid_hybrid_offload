"""Simulation configuration dataclasses.

All simulation parameters in one place, loadable from JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Dataset / arrival
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Parameters for synthetic request generation."""

    num_requests: int = 100
    prompt_length_dist: Literal["fixed", "uniform", "normal"] = "fixed"
    prompt_length_fixed: int = 512
    prompt_length_min: int = 64
    prompt_length_max: int = 2048
    output_length_dist: Literal["fixed", "uniform"] = "fixed"
    output_length_fixed: int = 256
    output_length_min: int = 32
    output_length_max: int = 1024

    # Controllable prefix overlap for cache hit rate experiments.
    # Fraction of each request's prompt that reuses the previous request's prefix.
    shared_prefix_ratio: float = 0.5
    shared_prefix_length: int | None = None  # explicit override (tokens)


@dataclass
class RequestArrivalConfig:
    """How requests arrive over time."""

    num_requests: int = 100
    arrival_pattern: Literal["burst", "poisson", "staggered"] = "poisson"
    poisson_rate: float = 1.0  # requests per second (real time)
    stagger_delay_steps: int = 5  # steps between staggered arrivals


@dataclass
class DatasetConfig:
    """Synthetic or real dataset configuration."""

    source: Literal["synthetic", "real"] = "synthetic"
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    real_dataset_path: str | None = None  # path to JSONL file


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeDecodeConfig:
    """Speculative decoding simulation parameters."""

    enabled: bool = True
    num_spec_tokens: int = 2  # K draft tokens per decode step

    # Acceptance mode
    accept_mode: Literal["fixed", "per_position"] = "per_position"
    acceptance_rate: float = 0.85  # used when accept_mode == "fixed"
    # Per-position end-to-end acceptance rate (measured on a real speculator):
    # P(draft[i] accepted) already encodes draft correctness + verification.
    acceptance_rates: list[float] | None = None  # e.g. [0.8, 0.7, 0.5, 0.3]


# ---------------------------------------------------------------------------
# GPU performance model
# ---------------------------------------------------------------------------


@dataclass
class GPUPerfConfig:
    """GPU performance model configuration.

    Model: latency_ms = a*loaded_tokens + b*computed_tokens
                        + c*loaded_tokens*computed_tokens + d

    Coefficients are fitted via least squares from data_points.
    """

    # User-provided data points: [(loaded, computed, latency_ms), ...]
    data_points: list[tuple[float, float, float]] | None = None

    # Explicit coefficient overrides (skip fitting if all set)
    loaded_coeff: float | None = None  # a
    computed_coeff: float | None = None  # b
    interaction_coeff: float | None = None  # c
    base_latency_ms: float | None = None  # d


# ---------------------------------------------------------------------------
# Top-level simulator config
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """Top-level configuration for a simulation run."""

    # Model
    model_name: str = "deepseek-ai/DeepSeek-V4-Flash"
    model_config_path: str | None = None  # path to config.json; uses defaults if None

    # Backend
    backend: Literal["vllm", "sglang"] = "vllm"

    # Sub-configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    arrival: RequestArrivalConfig = field(default_factory=RequestArrivalConfig)
    speculative: SpeculativeDecodeConfig = field(default_factory=SpeculativeDecodeConfig)
    gpu_perf: GPUPerfConfig = field(default_factory=GPUPerfConfig)

    # KV cache
    kv_cache_block_size: int = 16
    hash_block_size: int = 16
    max_model_len: int = 8192
    num_kv_cache_blocks: int = 4096

    # Model options
    use_fp4_indexer: bool = False  # deepseek_v4 indexer fp4 mode
    # SGLang DSV4 SWA/full token ratio (deepseek_v4_hook.py:57 overrides the
    # SGLang default to 0.1).  Real SGLang's own default is 0.8 for hybrid MLA
    # models without the DSV4 hook, so this MUST be configurable to stay
    # correct for non-DSV4 hybrid models.  vLLM ignores it (it has no
    # analogous knob — SWA sizing is implicit in the packed block pool).
    swa_full_tokens_ratio: float = 0.1

    # Execution
    warmup_steps: int = 10  # steps to exclude from metrics
    # Max consecutive zero-progress steps before the scheduler bails out with
    # a RuntimeError (decode alloc failure against a full KV pool with nothing
    # evictable would otherwise loop forever).  Tunable: lower for fast
    # iteration on OOM scenarios, higher for workloads with legitimately long
    # evict-wait stalls.  See scheduler._check_stall.
    stall_limit: int = 1000
    random_seed: int = 42
    output_dir: str | None = None
    verbose: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulatorConfig":
        """Load from a JSON config file."""
        with open(path) as f:
            data = json.load(f)

        # Parse sub-configs
        dataset_data = data.get("dataset", {})
        synthetic_data = dataset_data.get("synthetic", {})
        dataset = DatasetConfig(
            source=dataset_data.get("source", "synthetic"),
            real_dataset_path=dataset_data.get("real_dataset_path"),
            synthetic=SyntheticConfig(
                num_requests=synthetic_data.get("num_requests", 100),
                prompt_length_dist=synthetic_data.get("prompt_length_dist", "fixed"),
                prompt_length_fixed=synthetic_data.get("prompt_length_fixed", 512),
                prompt_length_min=synthetic_data.get("prompt_length_min", 64),
                prompt_length_max=synthetic_data.get("prompt_length_max", 2048),
                output_length_dist=synthetic_data.get("output_length_dist", "fixed"),
                output_length_fixed=synthetic_data.get("output_length_fixed", 256),
                output_length_min=synthetic_data.get("output_length_min", 32),
                output_length_max=synthetic_data.get("output_length_max", 1024),
                shared_prefix_ratio=synthetic_data.get("shared_prefix_ratio", 0.5),
                shared_prefix_length=synthetic_data.get("shared_prefix_length"),
            ),
        )

        spec_data = data.get("speculative", {})
        # Cross-validate enabled vs num_spec_tokens.  The CLI binds
        # ``enabled = num_spec_tokens > 0`` (run.py:43) — a single source of
        # truth — but the JSON path parses the two fields independently.  An
        # inconsistent JSON like ``{"enabled": false, "num_spec_tokens": 2}``
        # would otherwise inflate the KV pool (engine reads num_spec_tokens,
        # not enabled) while never running speculation — a silent ~2.3%
        # over-allocation with no drafts generated.  ``enabled`` is the
        # authoritative switch here (the user set it explicitly); when it is
        # False, force num_spec_tokens=0 so pool sizing and the spec engine
        # agree speculation is off.
        spec_enabled = spec_data.get("enabled", True)
        spec_num_tokens = spec_data.get("num_spec_tokens", 2)
        if not spec_enabled:
            spec_num_tokens = 0
        speculative = SpeculativeDecodeConfig(
            enabled=spec_enabled,
            num_spec_tokens=spec_num_tokens,
            accept_mode=spec_data.get("accept_mode", "per_position"),
            acceptance_rate=spec_data.get("acceptance_rate", 0.85),
            acceptance_rates=spec_data.get("acceptance_rates"),
        )

        gpu_data = data.get("gpu_perf", {})
        gpu_perf = GPUPerfConfig(
            data_points=gpu_data.get("data_points"),
            loaded_coeff=gpu_data.get("loaded_coeff"),
            computed_coeff=gpu_data.get("computed_coeff"),
            interaction_coeff=gpu_data.get("interaction_coeff"),
            base_latency_ms=gpu_data.get("base_latency_ms"),
        )

        arrival_data = data.get("arrival", {})
        arrival = RequestArrivalConfig(
            num_requests=arrival_data.get("num_requests", 100),
            arrival_pattern=arrival_data.get("arrival_pattern", "poisson"),
            poisson_rate=arrival_data.get("poisson_rate", 1.0),
            stagger_delay_steps=arrival_data.get("stagger_delay_steps", 5),
        )

        return cls(
            model_name=data.get("model_name", "deepseek-ai/DeepSeek-V4-Flash"),
            model_config_path=data.get("model_config_path"),
            backend=data.get("backend", "vllm"),
            dataset=dataset,
            arrival=arrival,
            speculative=speculative,
            gpu_perf=gpu_perf,
            use_fp4_indexer=data.get("use_fp4_indexer", False),
            swa_full_tokens_ratio=data.get("swa_full_tokens_ratio", 0.1),
            kv_cache_block_size=data.get("kv_cache_block_size", 16),
            hash_block_size=data.get("hash_block_size", 16),
            max_model_len=data.get("max_model_len", 8192),
            num_kv_cache_blocks=data.get("num_kv_cache_blocks", 4096),
            warmup_steps=data.get("warmup_steps", 10),
            stall_limit=data.get("stall_limit", 1000),
            random_seed=data.get("random_seed", 42),
            output_dir=data.get("output_dir"),
            verbose=data.get("verbose", False),
        )
