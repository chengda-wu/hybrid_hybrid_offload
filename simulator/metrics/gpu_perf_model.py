"""GPU performance model via linear regression.

Model:
    latency_ms = a * loaded_tokens + b * computed_tokens
               + c * interaction_tokens + d

where ``interaction_tokens`` is the per-request interaction mass.  For a
single request this is ``loaded * computed``; for a batch it is
``Σ(loaded_i * computed_i)`` (NOT ``(Σloaded) * (Σcomputed)``).  See
``predict`` for why.

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

    Per-step latency is capped at ``MAX_STEP_LATENCY_MS``: a single GPU
    forward pass on one device has a physical ceiling.  The cap is a safety
    net for inputs far outside the training envelope, NOT a load-bearing
    control on batch scaling — the interaction term is now additively
    decomposed per-request (see ``predict``), so a batch no longer snowballs
    as ``(Σloaded)·(Σcomputed)``.  The cap must sit above the largest
    training-point latency so the model can reproduce its own training data;
    raising it is safe precisely because the per-request decomposition
    removed the old O(N²) blow-up.
    """

    # Ceiling on a single step's predicted latency (ms).  Must be ≥ the
    # largest DEFAULT_DATA latency (130 ms) so the cap does not eat the
    # model's own training points.  See class docstring.
    MAX_STEP_LATENCY_MS: float = 200.0

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
        self._warned_negative = False
        self._warned_cap = False
        self._warned_extrapolation = False
        self._cap_count = 0
        self._cap_max_predicted = 0.0
        # Training-data envelope, set in _fit.  predict() warns (once) when an
        # input exceeds 2× these — the linear+interaction model is unreliable
        # far outside the fitted region (the c·m·n term dominates and the cap
        # then hides how detached from reality the raw prediction is).
        self._max_loaded: int = 0
        self._max_computed: int = 0
        self._fit()

    def _fit(self) -> None:
        """Fit coefficients for latency = a*m + b*n + c*m*n + d.

        Uses Gaussian elimination on the 4×4 normal equations (XᵀX)β = Xᵀy.
        """
        # Build data points
        if self._config.data_points:
            points = [PerfDataPoint(*p) for p in self._config.data_points]
        else:
            points = self.DEFAULT_DATA

        # Record the training envelope for extrapolation warning in predict().
        # Recorded BEFORE the explicit-coefficient early return below so that a
        # user who supplies coefficients (skipping the fit) still gets an
        # envelope: from their own data_points if provided, else DEFAULT_DATA
        # as a rough calibration-range proxy.  Pre-fix the early return left
        # _max_loaded/_max_computed at 0, and predict()'s ``> 0`` guard then
        # silenced the extrapolation warning permanently in coeff mode — even
        # for inputs wildly outside any sane range.  (A user who supplies
        # coefficients for a known range should also supply data_points to set
        # an accurate envelope; otherwise DEFAULT_DATA bounds the proxy.)
        if points:
            self._max_loaded = max(p.loaded_tokens for p in points)
            self._max_computed = max(p.computed_tokens for p in points)

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

        n = len(points)
        if n == 0:
            self._d = 1.0
            self._fitted = True
            return

        # Build XᵀX (4×4) and Xᵀy (4×1) for:
        #   y = a·m + b·n + c·m·n + d
        # columns: [m, n, m·n, 1]
        m = [p.loaded_tokens for p in points]
        n_ = [p.computed_tokens for p in points]
        mn = [p.loaded_tokens * p.computed_tokens for p in points]
        y = [p.latency_ms for p in points]

        # XᵀX
        s00 = sum(v * v for v in m)
        s01 = sum(m[i] * n_[i] for i in range(len(points)))
        s02 = sum(m[i] * mn[i] for i in range(len(points)))
        s03 = sum(m)
        s11 = sum(v * v for v in n_)
        s12 = sum(n_[i] * mn[i] for i in range(len(points)))
        s13 = sum(n_)
        s22 = sum(v * v for v in mn)
        s23 = sum(mn)
        s33 = float(len(points))

        # Xᵀy
        r0 = sum(m[i] * y[i] for i in range(len(points)))
        r1 = sum(n_[i] * y[i] for i in range(len(points)))
        r2 = sum(mn[i] * y[i] for i in range(len(points)))
        r3 = sum(y)

        # Gaussian elimination on augmented matrix [XᵀX | Xᵀy]
        A = [
            [s00, s01, s02, s03, r0],
            [s01, s11, s12, s13, r1],
            [s02, s12, s22, s23, r2],
            [s03, s13, s23, s33, r3],
        ]

        # Forward elimination
        for col in range(4):
            # Pivot
            pivot_row = max(range(col, 4), key=lambda r: abs(A[r][col]))
            if abs(A[pivot_row][col]) < 1e-12:
                continue  # singular column, leave as 0
            A[col], A[pivot_row] = A[pivot_row], A[col]
            pivot = A[col][col]
            for j in range(col, 5):
                A[col][j] /= pivot
            for row in range(4):
                if row != col and abs(A[row][col]) > 1e-15:
                    factor = A[row][col]
                    for j in range(col, 5):
                        A[row][j] -= factor * A[col][j]

        # Extract solution
        self._a = A[0][4]
        self._b = A[1][4]
        self._c = A[2][4]
        self._d = A[3][4]

        # Per-token costs should not be negative
        self._a = max(0.0, self._a)
        self._b = max(0.0, self._b)
        # c (interaction) and d (base) can legitimately be negative:
        # c < 0 when large batches amortize overhead; d < 0 when
        # data trends below the origin.  Clamping would distort
        # predictions, so they are left as-is.

        self._fitted = True

    def predict(
        self,
        loaded_tokens: int,
        computed_tokens: int,
        interaction_tokens: int | None = None,
    ) -> float:
        """Predict latency in milliseconds for a forward pass.

        Args:
            loaded_tokens: Total tokens already cached (Σ over the batch).
            computed_tokens: New tokens to compute in this forward (Σ over batch).
            interaction_tokens: Per-request interaction mass Σ(loaded_i *
                computed_i).  Defaults to ``loaded_tokens * computed_tokens``,
                which is correct for a single request.  Callers batching
                multiple requests MUST pass the per-request sum — multiplying
                the *batch totals* ``(Σloaded)·(Σcomputed)`` introduces
                phantom cross-request terms (request i's new tokens never
                attend request j's cache) and blows up as O(N²).  The
                per-request sum is the additively-correct decomposition: it
                equals ``loaded*computed`` for one request and grows linearly
                with batch size for identical requests.
        """
        assert self._fitted, "Model not fitted"
        if interaction_tokens is None:
            interaction_tokens = loaded_tokens * computed_tokens
        # Warn (once) on extrapolation far outside the training envelope.  The
        # interaction term grows unboundedly, so a prediction for e.g.
        # loaded=50000 (training max 8000) is detached from reality.  Surface
        # this so the user knows to add data points or interpret with caution.
        if not self._warned_extrapolation and (
            loaded_tokens > 2 * self._max_loaded
            or computed_tokens > 2 * self._max_computed
            or interaction_tokens > 2 * self._max_loaded * self._max_computed
        ) and (self._max_loaded > 0 or self._max_computed > 0):
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                "GPU perf model input loaded=%d computed=%d interaction=%d is "
                "far outside the training envelope (max loaded=%d, max "
                "computed=%d) — prediction is an extrapolation and may not "
                "reflect real latency.  Add data points covering this range or "
                "interpret with caution.  (This warning is printed once.)",
                loaded_tokens, computed_tokens, interaction_tokens,
                self._max_loaded, self._max_computed,
            )
            self._warned_extrapolation = True
        latency = (
            self._a * loaded_tokens
            + self._b * computed_tokens
            + self._c * interaction_tokens
            + self._d
        )
        if latency > self.MAX_STEP_LATENCY_MS:
            self._cap_count += 1
            if latency > self._cap_max_predicted:
                self._cap_max_predicted = latency
            if not self._warned_cap:
                import logging
                _log = logging.getLogger(__name__)
                _log.warning(
                    "GPU perf model predicted latency %.2f ms (> cap %.1f ms) "
                    "for loaded=%d computed=%d interaction=%d — clamped to cap.  "
                    "The cap is a safety net for inputs far outside the training "
                    "envelope, modeling the physical ceiling of a single GPU "
                    "forward pass.  (Further clamps are counted silently; final "
                    "count reported at end.)",
                    latency, self.MAX_STEP_LATENCY_MS,
                    loaded_tokens, computed_tokens, interaction_tokens,
                )
                self._warned_cap = True
            latency = self.MAX_STEP_LATENCY_MS
        if latency < 0:
            if not self._warned_negative:
                import logging
                _log = logging.getLogger(__name__)
                _log.warning(
                    "GPU perf model predicted negative latency (%.4f ms) for "
                    "loaded=%d computed=%d — floor to 0.  Consider adding data "
                    "points near the origin (e.g. [0, 1, <latency>]) to anchor "
                    "the fit.  (This warning is printed once.)",
                    latency, loaded_tokens, computed_tokens,
                )
                self._warned_negative = True
            latency = 0.0
        return latency

    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        """Return (a, b, c, d)."""
        return (self._a, self._b, self._c, self._d)

    @property
    def cap_stats(self) -> tuple[int, float]:
        """Return (num_clamped_steps, max_predicted_latency_ms_before_clamp).

        For end-of-run reporting: the first clamp warns once (see predict),
        and these counters give the aggregate picture without per-step spam.
        """
        return (self._cap_count, self._cap_max_predicted)
