"""SimulationEngine — wires everything together and runs the simulation."""

from __future__ import annotations

from simulator.config.model_config import (
    KVBackendConfig,
    ModelArchitecture,
    SGLangConfig,
    VLLMConfig,
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

        # Build backend config
        self._backend_config = KVBackendConfig(
            model_arch=self._model_arch,
            block_size=config.kv_cache_block_size,
            hash_block_size=config.hash_block_size,
            max_model_len=config.max_model_len,
            num_kv_cache_blocks=config.num_kv_cache_blocks,
            scheduler_block_size=config.kv_cache_block_size,
            page_size=1,  # SGLang token-level
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
        # Load data
        loader = DatasetLoader(self._config.dataset, seed=self._config.random_seed)
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
        )

    def _build_backend(self) -> KVBackend:
        if self._config.backend == "vllm":
            from simulator.kv_cache.vllm_backend import vLLMBackend

            return vLLMBackend(self._backend_config)
        else:
            from simulator.kv_cache.sglang_backend import SGLangBackend

            return SGLangBackend(self._backend_config)
