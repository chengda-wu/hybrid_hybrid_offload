#!/usr/bin/env python3
"""KV Cache Simulation System — CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulator.config.simulator_config import (
    DatasetConfig,
    GPUPerfConfig,
    SimulatorConfig,
    SpeculativeDecodeConfig,
    SyntheticConfig,
)
from simulator.core.engine import SimulationEngine


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = SimulatorConfig.from_json(args.config)
    else:
        config = SimulatorConfig(
            backend=args.backend,
            dataset=DatasetConfig(
                synthetic=SyntheticConfig(
                    num_requests=args.num_requests,
                    prompt_length_fixed=args.prompt_length,
                    prompt_length_dist="fixed",
                    output_length_fixed=args.output_length,
                    output_length_dist="fixed",
                    shared_prefix_ratio=args.shared_prefix_ratio,
                )
            ),
            speculative=SpeculativeDecodeConfig(
                enabled=args.num_spec_tokens > 0,
                num_spec_tokens=args.num_spec_tokens,
                accept_mode=args.accept_mode,
                acceptance_rate=args.acceptance_rate,
                acceptance_rates=args.acceptance_rates,
                draft_accuracy=args.draft_accuracy,
            ),
            model_config_path=args.model_config,
            kv_cache_block_size=args.kv_block_size,
            max_model_len=args.max_model_len,
            num_kv_cache_blocks=args.num_kv_blocks,
            random_seed=args.seed,
            verbose=args.verbose,
        )
        if args.gpu_data_points:
            config.gpu_perf = GPUPerfConfig(data_points=args.gpu_data_points)

    engine = SimulationEngine(config)
    report = engine.run()

    output = report.to_json()
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KV Cache Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (optional — overrides CLI args)
    p.add_argument("--config", type=str, help="Path to JSON config file")

    # Backend
    p.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")

    # Dataset
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--prompt-length", type=int, default=512)
    p.add_argument("--output-length", type=int, default=256)
    p.add_argument("--shared-prefix-ratio", type=float, default=0.5)

    # Speculative decode
    p.add_argument("--num-spec-tokens", type=int, default=2)
    p.add_argument("--accept-mode", choices=["fixed", "per_position"], default="per_position")
    p.add_argument("--acceptance-rate", type=float, default=0.85)
    p.add_argument(
        "--acceptance-rates", type=float, nargs="+",
        help="Per-position acceptance rates, e.g. 0.8 0.7 0.5 0.3"
    )
    p.add_argument("--draft-accuracy", type=float, default=0.7)

    # GPU perf
    p.add_argument(
        "--gpu-data-points", type=str,
        help="JSON list of [loaded, computed, latency_ms] triples"
    )

    # KV cache
    p.add_argument("--kv-block-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--num-kv-blocks", type=int, default=4096)

    # Model
    p.add_argument("--model-config", type=str, help="Path to HF config.json")

    # Execution
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", "-o", type=str, help="Output JSON file")
    p.add_argument("--verbose", "-v", action="store_true")

    return p


if __name__ == "__main__":
    sys.exit(main())
