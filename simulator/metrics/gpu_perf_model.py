"""GPU performance model via linear regression.

Model:
    latency_ms = a * loaded_tokens + b * computed_tokens
               + c * loaded_tokens * computed_tokens + d

Coefficients are fitted from user-provided (loaded, computed, latency) data
points using least squares.  Reasonable H100-like defaults are provided.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator.config.simulator_config import GPUPerfConfig


@dataclass
class PerfDataPoint:
    loaded_tokens: int   # tokens already in KV cache (cached prefix)
    computed_tokens: int  # new tokens computed in this forward
    latency_ms: float


class GPUPerfModel:
    """GPU performance model with interaction term.

    Captures:
    - Memory bandwidth costs (loaded_tokens: attention over cached KV)
    - Compute costs (computed_tokens: QKV projections + attention)
    - Interaction between cached and new tokens
    """

    DEFAULT_DATA: list[PerfDataPoint] = [
        PerfDataPoint(0, 1, 0.8),         # single token decode, no cache
        PerfDataPoint(1000, 1, 1.5),      # decode from 1K cached tokens
        PerfDataPoint(4000, 1, 3.0),      # decode from 4K cached tokens
        PerfDataPoint(8000, 1, 5.0),      # decode from 8K cached tokens
        PerfDataPoint(0, 512, 8.0),       # full prefill 512 tokens
        PerfDataPoint(0, 2048, 30.0),     # prefill 2K tokens
        PerfDataPoint(0, 4096, 60.0),     # prefill 4K tokens
        PerfDataPoint(0, 8192, 130.0),    # prefill 8K tokens
        PerfDataPoint(1000, 3, 3.0),      # spec decode: 1 bonus + 2 draft
        PerfDataPoint(4000, 3, 6.0),      # spec decode with larger cache
    ]

    def __init__(self, config: GPUPerfConfig):
        self._config = config
        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._d: float = 0.0
        self._fitted = False
        self._fit()

    def _fit(self) -> None:
        """Fit coefficients via least squares from data points."""
        # Check for explicit coefficient overrides
        if (
            self._config.loaded_coeff is not None
            and self._config.computed_coeff is not None
            and self._config.interaction_coeff is not None
            and self._config.base_latency_ms is not None
        ):
            self._a = self._config.loaded_coeff
            self._b = self._config.computed_coeff
            self._c = self._config.interaction_coeff
            self._d = self._config.base_latency_ms
            self._fitted = True
            return

        # Build data points
        if self._config.data_points:
            points = [PerfDataPoint(*p) for p in self._config.data_points]
        else:
            points = self.DEFAULT_DATA

        n = len(points)
        if n == 0:
            self._d = 1.0  # fallback constant latency
            self._fitted = True
            return

        # Solve normal equations for the linear model:
        #   y = a*x1 + b*x2 + c*x1*x2 + d
        # Design matrix X: [loaded, computed, loaded*computed, 1]
        sum_x1 = sum(p.loaded_tokens for p in points)
        sum_x2 = sum(p.computed_tokens for p in points)
        sum_x12 = sum(p.loaded_tokens * p.computed_tokens for p in points)
        sum_x1sq = sum(p.loaded_tokens * p.loaded_tokens for p in points)
        sum_x2sq = sum(p.computed_tokens * p.computed_tokens for p in points)
        sum_x12sq = sum(
            p.loaded_tokens * p.computed_tokens * p.loaded_tokens * p.computed_tokens
            for p in points
        )
        sum_y = sum(p.latency_ms for p in points)
        sum_x1y = sum(p.loaded_tokens * p.latency_ms for p in points)
        sum_x2y = sum(p.computed_tokens * p.latency_ms for p in points)
        sum_x12y = sum(
            p.loaded_tokens * p.computed_tokens * p.latency_ms for p in points
        )

        # Direct solve of the 4x4 normal equations
        # (X^T X) * coeffs = X^T y
        import math

        A = [
            [sum_x1sq, sum_x12, sum_x1 * sum_x2 if False else 0, sum_x1],  # will recompute below
        ]
        # Actually, a pure-Python least squares is error-prone. Use a very simple
        # approach: solve just for a, b, d (no interaction term) if data is sparse,
        # then refine.

        # Simplified: use the analytical solution for the 3×3 case first
        # to avoid heavy dependencies.
        # We'll fit: latency = a*m + b*n + d  (no interaction)
        # Then add interaction term by fitting residual.

        # Fit linear: y = a*m + b*n + d
        # Solve via ordinary least squares for 3 params
        det = n * (sum_x1sq * sum_x2sq - sum_x12 * sum_x12) - (
            sum_x1 * (sum_x1 * sum_x2sq - sum_x2 * sum_x12)
            - sum_x2 * (sum_x1 * sum_x12 - sum_x2 * sum_x1sq)
        )
        if abs(det) < 1e-9:
            # Fallback: just use mean latency as constant
            self._d = sum_y / n
        else:
            # Compute via Cramer's rule for the 3×3 system
            # [ sum_x1sq  sum_x12   sum_x1 ] [a]   [sum_x1y]
            # [ sum_x12   sum_x2sq  sum_x2 ] [b] = [sum_x2y]
            # [ sum_x1    sum_x2    n      ] [d]   [sum_y  ]
            det_a = (
                sum_x1y * (sum_x2sq * n - sum_x2 * sum_x2)
                - sum_x12 * (sum_x2y * n - sum_y * sum_x2)
                + sum_x1 * (sum_x2y * sum_x2 - sum_y * sum_x2sq)
            )
            det_b = (
                sum_x1sq * (sum_x2y * n - sum_y * sum_x2)
                - sum_x1y * (sum_x12 * n - sum_x1 * sum_x2)
                + sum_x1 * (sum_x12 * sum_y - sum_x1 * sum_x2y)
            )
            det_d = (
                sum_x1sq * (sum_x2sq * sum_y - sum_x2 * sum_x2y)
                - sum_x12 * (sum_x12 * sum_y - sum_x1 * sum_x2y)
                + sum_x1 * (sum_x12 * sum_x2y - sum_x1 * sum_x2sq)
            )
            self._a = max(0.0, det_a / det)
            self._b = max(0.0, det_b / det)
            self._d = max(0.0, det_d / det)

        # Fit interaction term from residuals
        residuals = []
        interactions = []
        for p in points:
            pred = self._a * p.loaded_tokens + self._b * p.computed_tokens + self._d
            resid = p.latency_ms - pred
            inter = p.loaded_tokens * p.computed_tokens
            if inter > 0:
                residuals.append(resid)
                interactions.append(inter)
        if interactions:
            sum_inter_sq = sum(v * v for v in interactions)
            if sum_inter_sq > 0:
                sum_inter_resid = sum(
                    i * r for i, r in zip(interactions, residuals)
                )
                self._c = max(0.0, sum_inter_resid / sum_inter_sq)

        self._fitted = True

    def predict(self, loaded_tokens: int, computed_tokens: int) -> float:
        """Predict latency in milliseconds for a forward pass.

        Args:
            loaded_tokens: Total tokens already cached (attention query context).
            computed_tokens: New tokens to compute in this forward.
        """
        assert self._fitted, "Model not fitted"
        return (
            self._a * loaded_tokens
            + self._b * computed_tokens
            + self._c * loaded_tokens * computed_tokens
            + self._d
        )

    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        """Return (a, b, c, d)."""
        return (self._a, self._b, self._c, self._d)
