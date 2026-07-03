"""SimulationEngine — wires everything together and runs the simulation."""

from __future__ import annotations

from simulator.config.model_config import (
    KVBackendConfig,
    ModelArchitecture,
)
from simulator.config.simulator_config import SimulatorConfig
from simulator.core.request_state import SimRequestState
from simulator.core.scheduler import SimulatorScheduler
from simulator.data.dataset_loader import DatasetLoader
from simulator.kv_cache.base import KVBackend
from simulator.metrics.gpu_perf_model import GPUPerfModel
from simulator.metrics.recorder import MetricsRecorder
from simulator.metrics.stats import SimulationReport, StatisticsComputer
from simulator.speculative.acceptance import AcceptanceModel


class SimulationEngine:
    """Top-level simulation runner."""

    def __init__(self, config: SimulatorConfig):
        self._config = config

        # Build model architecture
        if config.model_config_path:
            self._model_arch = ModelArchitecture.from_json(config.model_config_path)
        else:
            self._model_arch = ModelArchitecture.deepseek_v4_flash()
        self._model_arch.use_fp4_indexer = config.use_fp4_indexer

        # Build backend config.
        # For hybrid models, compute scheduler_block_size as LCM of group block sizes
        # and hash_block_size as GCD (required by vLLM assertion).
        import math

        if self._model_arch.is_mla and self._model_arch.compress_ratios:
            group_block_sizes = [g[1] for g in self._model_arch.layer_groups]
            scheduler_block_size_val = group_block_sizes[0]
            hash_block_size_val = group_block_sizes[0]
            for bs in group_block_sizes[1:]:
                scheduler_block_size_val = (
                    scheduler_block_size_val * bs // math.gcd(scheduler_block_size_val, bs)
                )
                hash_block_size_val = math.gcd(hash_block_size_val, bs)
            main_block_size = max(group_block_sizes)
        else:
            # Non-hybrid: use config values directly
            main_block_size = config.kv_cache_block_size
            hash_block_size_val = config.kv_cache_block_size
            scheduler_block_size_val = config.kv_cache_block_size

        self._main_block_size = main_block_size
        self._backend_config = KVBackendConfig(
            model_arch=self._model_arch,
            block_size=main_block_size,
            hash_block_size=hash_block_size_val,
            max_model_len=config.max_model_len,
            num_kv_cache_blocks=config.num_kv_cache_blocks,
            scheduler_block_size=scheduler_block_size_val,
            num_spec_tokens=config.speculative.num_spec_tokens,
        )

        # Build components
        self._backend = self._build_backend()
        self._acceptance = AcceptanceModel(config.speculative, config.random_seed)
        self._gpu_perf = GPUPerfModel(config.gpu_perf)
        self._recorder = MetricsRecorder()

        self._scheduler = SimulatorScheduler(
            config=config,
            kv_backend=self._backend,
            acceptance_model=self._acceptance,
            gpu_perf_model=self._gpu_perf,
            recorder=self._recorder,
        )

    def run(self) -> SimulationReport:
        """Run the full simulation and return the report."""
        # Print config summary
        kv_size_bytes = self._backend.total_bytes
        kv_size_gb = kv_size_bytes / (1024**3)
        print(
            f"Backend: {self._backend.name} | "
            f"Model: {self._model_arch.model_type} ({self._model_arch.num_layers} layers) | "
            f"KV Cache: {kv_size_gb:.2f} GB ({self._config.num_kv_cache_blocks} blocks × "
            f"{self._main_block_size} tokens)"
        )
        print(
            f"Requests: {self._config.dataset.synthetic.num_requests} | "
            f"Spec tokens: K={self._config.speculative.num_spec_tokens} | "
            f"Seed: {self._config.random_seed}"
        )

        # Load data
        loader = DatasetLoader(
            self._config.dataset, seed=self._config.random_seed,
            arrival_config=self._config.arrival,
        )
        request_datas = loader.load()

        # Build SimRequestStates
        requests = []
        for rd in request_datas:
            sim_req = self._backend.create_request(
                rd.request_id, rd.prompt_token_ids, len(rd.ground_truth_output)
            )
            state = SimRequestState(
                request_id=rd.request_id,
                prompt_token_ids=list(rd.prompt_token_ids),
                ground_truth_output=list(rd.ground_truth_output),
                max_output_tokens=len(rd.ground_truth_output),
                arrival_time=rd.arrival_time,
                backend_req=sim_req,
            )
            requests.append(state)

        self._scheduler.load(requests)

        # Main loop
        while self._scheduler.step():
            pass

        # Compute and return report
        stats = StatisticsComputer()
        return stats.compute(
            recorder=self._recorder,
            backend=self._backend.name,
            kv_cache_size_gb=kv_size_gb,
        )

    def _build_backend(self) -> KVBackend:
        if self._config.backend == "vllm":
            from simulator.kv_cache.vllm_backend import vLLMBackend

            return vLLMBackend(self._backend_config)
        else:
            from simulator.kv_cache.sglang_backend import SGLangBackend

            return SGLangBackend(self._backend_config,
                                 num_spec_tokens=self._config.speculative.num_spec_tokens)
