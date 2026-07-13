#!/usr/bin/env python3
"""KV Cache Simulation System — CLI entry point.

CLI flag names are the kebab-case of the JSON/``SimulatorConfig`` field names
(``--kv-cache-block-size`` ↔ ``kv_cache_block_size`` etc.), so a flag maps 1:1
to the JSON key of the same field.  In ``--config`` mode, any CLI flag that
differs from its default overrides the JSON value — the same fields are
overridable in both modes (no special-casing like the old ``--fp4-indexer``-
only override).
"""

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
        config = SimulatorConfig()

    # Apply CLI overrides.  In --config mode only non-default flags override
    # the JSON value; in from-scratch mode every flag is set (defaults fill the
    # gaps), so the same logic produces the full config.  Dataset/spec sub-
    # configs are only built from-scratch (no JSON-sub-config CLI flags), so
    # they are skipped in --config mode.
    if not args.config:
        config.dataset = DatasetConfig(
            synthetic=SyntheticConfig(
                num_requests=args.num_requests,
                prompt_length_fixed=args.prompt_length,
                prompt_length_dist="fixed",
                output_length_fixed=args.output_length,
                output_length_dist="fixed",
                shared_prefix_ratio=args.shared_prefix_ratio,
            )
        )
        config.speculative = SpeculativeDecodeConfig(
            enabled=args.num_spec_tokens > 0,
            num_spec_tokens=args.num_spec_tokens,
            accept_mode=args.accept_mode,
            acceptance_rate=args.acceptance_rate,
            acceptance_rates=args.acceptance_rates,
        )
        if args.gpu_data_points is not None:
            config.gpu_perf = GPUPerfConfig(data_points=args.gpu_data_points)

    # Scalar fields: override the JSON value whenever the CLI flag was set to a
    # non-default value.  Checking against the parser default (not None) lets
    # --config users tweak any single knob from the command line.
    _override(config, "backend", args.backend, "vllm")
    _override(config, "model_config_path", args.model_config_path, None)
    _override(config, "use_fp4_indexer", args.use_fp4_indexer, False)
    _override(config, "swa_full_tokens_ratio", args.swa_full_tokens_ratio, 0.1)
    _override(config, "kv_cache_block_size", args.kv_cache_block_size, 16)
    _override(config, "max_model_len", args.max_model_len, 8192)
    _override(config, "num_kv_cache_blocks", args.num_kv_cache_blocks, 4096)
    _override(config, "random_seed", args.random_seed, 42)
    _override(config, "stall_limit", args.stall_limit, 1000)
    _override(config, "verbose", args.verbose, False)

    engine = SimulationEngine(config)
    report = engine.run()

    output = report.to_json()
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


def _override(config: SimulatorConfig, field: str, cli_value, default) -> None:
    """Set ``config.<field>`` to ``cli_value`` when it differs from default.

    In --config mode this applies a CLI override only when the user passed a
    # non-default value (so a bare ``--config x.json`` doesn't clobber JSON
    # with parser defaults).  In from-scratch mode every flag is set, so this
    # also builds the full config from defaults.
    """
    if cli_value != default:
        setattr(config, field, cli_value)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KV Cache Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Disallow unambiguous-prefix abbreviation (e.g. --model-config as a
        # shorthand for --model-config-path).  Flag names mirror JSON field
        # names 1:1; abbreviations would reintroduce the exact naming
        # inconsistency this cleanup removed.
        allow_abbrev=False,
    )

    # Config file (optional — CLI flags override JSON values when set)
    p.add_argument("--config", type=str, help="Path to JSON config file")

    # Backend
    p.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")

    # Dataset
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--prompt-length", type=int, default=512)
    p.add_argument("--output-length", type=int, default=512)
    p.add_argument("--shared-prefix-ratio", type=float, default=0.5)

    # Speculative decode
    p.add_argument("--num-spec-tokens", type=int, default=2)
    p.add_argument("--accept-mode", choices=["fixed", "per_position"], default="per_position")
    p.add_argument("--acceptance-rate", type=float, default=0.85)
    p.add_argument(
        "--acceptance-rates", type=float, nargs="+",
        help="Per-position acceptance rates, e.g. 0.8 0.7 0.5 0.3"
    )

    # GPU perf
    p.add_argument(
        "--gpu-data-points", type=str,
        help="JSON list of [loaded, computed, latency_ms] triples"
    )

    # KV cache (flag = kebab-case of the JSON field name)
    p.add_argument("--kv-cache-block-size", type=int, default=16,
                   help="KV cache block size (tokens per block). JSON: kv_cache_block_size.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="JSON: max_model_len.")
    p.add_argument("--num-kv-cache-blocks", type=int, default=4096,
                   help="Total KV cache block pool size. JSON: num_kv_cache_blocks.")

    # Model (flag = kebab-case of the JSON field name)
    p.add_argument("--model-config-path", type=str,
                   help="Path to HF config.json. JSON: model_config_path.")
    p.add_argument("--use-fp4-indexer", action="store_true",
                   help="Use fp4 indexer (68 B/token instead of 132). JSON: use_fp4_indexer.")
    p.add_argument(
        "--swa-full-tokens-ratio", type=float, default=0.1,
        help="SGLang DSV4 SWA/full token ratio (deepseek_v4_hook.py:57 default "
             "0.1). Raise for a larger SWA pool. vLLM ignores this. "
             "JSON: swa_full_tokens_ratio.",
    )

    # Execution (flag = kebab-case of the JSON field name)
    p.add_argument("--random-seed", type=int, default=42,
                   help="JSON: random_seed.")
    p.add_argument(
        "--stall-limit", type=int, default=1000,
        help="Max consecutive zero-progress steps before the scheduler errors "
             "out (prevents infinite loop on KV-pool-full deadlock). Default 1000."
    )
    p.add_argument("--output", "-o", type=str, help="Output JSON file")
    p.add_argument("--verbose", "-v", action="store_true")

    return p


if __name__ == "__main__":
    sys.exit(main())
