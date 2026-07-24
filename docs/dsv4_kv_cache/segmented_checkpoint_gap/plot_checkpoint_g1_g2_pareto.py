#!/usr/bin/env python3
"""Build the DSV4 Flash two-gap checkpoint Pareto experiment.

The experiment compares every candidate against the same uniformly sampled
prefix-length workload. Mean storage is evaluated exactly over the complete
integer interval, while recovery compute uses nested common samples and the
exact discrete formula from segmented-checkpoint-gap-analysis.md.

The search has three stages:

1. a coarse gap grid over every split layer;
2. a medium-resolution integer neighborhood around the near-frontier;
3. a final integer neighborhood evaluated with the largest common N sample.

Only mean DRAM checkpoint storage and mean recovery token-layers participate in
Pareto dominance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np


L = 43
W = 128
D = W - 1
GIB = 1024**3

LAYER_CHECKPOINT_BYTES = []
for _ in range(20):
    LAYER_CHECKPOINT_BYTES.extend([157_824, 600_192])
LAYER_CHECKPOINT_BYTES.append(157_824)
LAYER_CHECKPOINT_BYTES.extend([74_880, 74_880])
PREFIX_BYTES = np.concatenate(
    ([0], np.cumsum(LAYER_CHECKPOINT_BYTES, dtype=np.int64))
)
TOTAL_BYTES = int(PREFIX_BYTES[-1])

# Layer indices increase from the input side toward the output side.  Therefore
# split=l is deeper when l is larger.  Layers 1..l, farther from the output, use
# g2; layers l+1..L, nearer the output, use g1.  Recovery walks from the output
# toward the input and therefore still evaluates g1 before g2.

# These are the preceding run's regression guards remapped from the old
# output-to-input layer count via l_new = L - l_old.
PREVIOUS_REPRESENTATIVE_PARAMETERS = [
    (6, 6122, 6126),
    (2, 2727, 2779),
    (1, 1394, 1422),
    (1, 689, 719),
    (1, 341, 364),
    (1, 170, 171),
    (1, 85, 85),
    (1, 43, 43),
    (1, 22, 22),
]

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "dsv4_checkpoint_g1_g2_pareto.csv"
DEFAULT_JSON = HERE / "dsv4_checkpoint_g1_g2_metadata.json"
DEFAULT_PNG = HERE / "imgs" / "dsv4-checkpoint-g1-g2-pareto.png"


def make_nested_samples(
    n_min: int,
    n_max: int,
    counts: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """Return nested common random samples, always including both endpoints."""
    if n_min >= n_max:
        raise ValueError("n_min must be smaller than n_max")
    if not counts or min(counts) < 2:
        raise ValueError("every sample count must be at least 2")
    population = n_max - n_min + 1
    counts = sorted(set(min(int(value), population) for value in counts))
    largest = counts[-1]
    rng = np.random.default_rng(seed)
    interior_count = max(0, largest - 2)
    interior = rng.choice(
        n_max - n_min - 1,
        size=interior_count,
        replace=False,
    ).astype(np.int64) + n_min + 1
    rng.shuffle(interior)
    result: dict[int, np.ndarray] = {}
    for count in counts:
        values = np.concatenate(
            (
                np.asarray([n_min, n_max], dtype=np.int64),
                interior[: max(0, count - 2)],
            )
        )
        result[count] = np.sort(values)
    return result


def make_gap_grid(max_gap: int, dense_max: int, coarse_step: int) -> np.ndarray:
    if max_gap < 1 or dense_max < 1 or coarse_step < 1:
        raise ValueError("gap bounds and step must be positive")
    dense_max = min(dense_max, max_gap)
    dense = np.arange(1, dense_max + 1, dtype=np.int64)
    first_coarse = ((dense_max // coarse_step) + 1) * coarse_step
    coarse = np.arange(first_coarse, max_gap + 1, coarse_step, dtype=np.int64)
    boundaries = np.asarray(
        [height * D + 1 for height in range(1, L)], dtype=np.int64
    )
    boundaries = boundaries[boundaries <= max_gap]
    powers = np.asarray(
        [1 << power for power in range(max_gap.bit_length()) if (1 << power) <= max_gap],
        dtype=np.int64,
    )
    return np.unique(
        np.concatenate((dense, coarse, boundaries, powers, [max_gap]))
    )


def checkpoint_count_prefix(x: int, gaps: np.ndarray) -> np.ndarray:
    """F_g(x) = sum_{N=1}^x ceil(N/g), vectorized over gaps."""
    if x <= 0:
        return np.zeros_like(gaps, dtype=np.float64)
    q = x // gaps
    r = x % gaps
    return gaps * q * (q + 1) // 2 + (q + 1) * r


def exact_mean_checkpoint_counts(
    gaps: np.ndarray, n_min: int, n_max: int
) -> np.ndarray:
    count = n_max - n_min + 1
    totals = checkpoint_count_prefix(n_max, gaps) - checkpoint_count_prefix(
        n_min - 1, gaps
    )
    return totals.astype(np.float64) / count


def triangle_removed(phase: np.ndarray, height: int) -> np.ndarray:
    """Exact T_h(r) for an integer scalar/array phase."""
    phase = np.asarray(phase, dtype=np.int64)
    m = np.maximum((phase - 1) // D, 0)
    m = np.minimum(m, height)
    return m * phase - D * m * (m + 1) // 2


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Return nondominated rows [storage, compute, g1, g2, l]."""
    if points.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    order = np.lexsort(
        (
            points[:, 4],
            points[:, 3],
            points[:, 2],
            points[:, 1],
            points[:, 0],
        )
    )
    ordered = points[order]
    keep = np.zeros(ordered.shape[0], dtype=bool)
    best_compute = np.inf
    for index, compute in enumerate(ordered[:, 1]):
        if compute < best_compute - 1e-10:
            keep[index] = True
            best_compute = compute
    return ordered[keep]


def near_frontier(
    points: np.ndarray, frontier: np.ndarray, relative_margin: float
) -> np.ndarray:
    """Keep points within a relative compute margin of the global frontier."""
    if points.size == 0 or frontier.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    frontier = frontier[np.argsort(frontier[:, 0])]
    indices = np.searchsorted(frontier[:, 0], points[:, 0], side="right") - 1
    valid = indices >= 0
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[valid] = points[valid, 1] <= (
        frontier[indices[valid], 1] * (1.0 + relative_margin)
    )
    return points[keep]


def group_candidates(candidates: set[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for split, g1, g2 in candidates:
        grouped[(split, g1)].append(g2)
    for values in grouped.values():
        values.sort()
    return grouped


def candidate_neighborhood(
    seeds: np.ndarray,
    max_gap: int,
    radius: int,
    stride: int,
    include_full_axes: bool,
) -> set[tuple[int, int, int]]:
    """Generate bounded g1/g2 neighborhoods around seed parameter rows."""
    offsets = sorted(set(range(-radius, radius + 1, stride)) | {0})
    candidates: set[tuple[int, int, int]] = set()
    for _, _, raw_g1, raw_g2, raw_split in seeds:
        split = int(raw_split)
        base_g1 = int(raw_g1)
        base_g2 = int(raw_g2)
        for delta1 in offsets:
            g1 = base_g1 + delta1
            if not 1 <= g1 <= max_gap:
                continue
            for delta2 in offsets:
                g2 = base_g2 + delta2
                if g1 <= g2 <= max_gap:
                    candidates.add((split, g1, g2))
        if include_full_axes:
            for delta in range(-radius, radius + 1):
                g1 = base_g1 + delta
                if 1 <= g1 <= min(base_g2, max_gap):
                    candidates.add((split, g1, base_g2))
                g2 = base_g2 + delta
                if base_g1 <= g2 <= max_gap:
                    candidates.add((split, base_g1, g2))
        candidates.add((split, base_g1, base_g2))
    return candidates


class Evaluator:
    def __init__(
        self,
        ns: np.ndarray,
        n_min: int,
        n_max: int,
        storage_limit_gib: float,
        block_size: int,
        workers: int,
    ) -> None:
        self.ns = ns
        self.n_min = n_min
        self.n_max = n_max
        self.storage_limit_gib = storage_limit_gib
        self.block_size = block_size
        self.workers = workers
        self._mean_count: dict[int, float] = {}
        self._phase1: dict[int, np.ndarray] = {}

    def prepare_gaps(self, gaps: set[int] | np.ndarray) -> None:
        missing = sorted(int(gap) for gap in gaps if int(gap) not in self._mean_count)
        if not missing:
            return
        values = np.asarray(missing, dtype=np.int64)
        means = exact_mean_checkpoint_counts(values, self.n_min, self.n_max)
        for gap, mean in zip(missing, means, strict=True):
            self._mean_count[gap] = float(mean)

    def phase1(self, gap: int) -> np.ndarray:
        value = self._phase1.get(gap)
        if value is None:
            value = self.ns % gap
            self._phase1[gap] = value
        return value

    def evaluate_group(self, split: int, g1: int, g2_values: list[int]) -> np.ndarray:
        if not g2_values:
            return np.empty((0, 5), dtype=np.float64)
        g2 = np.asarray(g2_values, dtype=np.int64)
        far_bytes = int(PREFIX_BYTES[split])
        near_bytes = TOTAL_BYTES - far_bytes
        storage = (
            near_bytes * self._mean_count[g1]
            + far_bytes * np.asarray([self._mean_count[int(value)] for value in g2])
        ) / GIB
        valid = storage <= self.storage_limit_gib
        if not np.any(valid):
            return np.empty((0, 5), dtype=np.float64)
        g2 = g2[valid]
        storage = storage[valid]

        near_height = L - split
        r1 = self.phase1(g1)
        near_compute = float(
            np.mean(near_height * r1 - triangle_removed(r1, near_height))
        )
        offset1 = np.minimum(r1, near_height * D)
        boundary1 = self.ns - offset1
        rows: list[np.ndarray] = []
        for start in range(0, g2.size, self.block_size):
            stop = min(start + self.block_size, g2.size)
            block = g2[start:stop]
            phase2 = boundary1[:, None] % block[None, :]
            base2 = offset1[:, None] + phase2
            far_compute = (
                split * base2 - triangle_removed(phase2, split)
            ).mean(axis=0)
            compute = near_compute + far_compute
            rows.append(
                np.column_stack(
                    (
                        storage[start:stop],
                        compute,
                        np.full(block.size, g1),
                        block,
                        np.full(block.size, split),
                    )
                )
            )
        return np.concatenate(rows, axis=0)

    def evaluate_grouped(
        self, grouped: dict[tuple[int, int], list[int]]
    ) -> np.ndarray:
        all_gaps = {g1 for _, g1 in grouped}
        for values in grouped.values():
            all_gaps.update(values)
        self.prepare_gaps(all_gaps)
        items = list(grouped.items())

        def run(item: tuple[tuple[int, int], list[int]]) -> np.ndarray:
            (split, g1), g2_values = item
            return self.evaluate_group(split, g1, g2_values)

        chunks: list[np.ndarray] = []
        if self.workers <= 1:
            for item in items:
                value = run(item)
                if value.size:
                    chunks.append(value)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for value in executor.map(run, items):
                    if value.size:
                        chunks.append(value)
        if not chunks:
            return np.empty((0, 5), dtype=np.float64)
        return np.concatenate(chunks, axis=0)


def evaluate_coarse_grid(
    evaluator: Evaluator,
    gaps: np.ndarray,
    cloud_points_per_split: int,
    cloud_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    evaluator.prepare_gaps(gaps)
    local_frontiers: list[np.ndarray] = []
    all_local_points: list[np.ndarray] = []
    cloud_chunks: list[np.ndarray] = []
    candidate_count = 0
    gap_values = [int(value) for value in gaps]
    rng = np.random.default_rng(cloud_seed)
    for split in range(1, L):
        grouped = {
            (split, g1): gap_values[index:]
            for index, g1 in enumerate(gap_values)
        }
        points = evaluator.evaluate_grouped(grouped)
        candidate_count += points.shape[0]
        local = pareto_frontier(points)
        local_frontiers.append(local)
        all_local_points.append(local)
        if cloud_points_per_split > 0 and points.shape[0] > 0:
            count = min(cloud_points_per_split, points.shape[0])
            indices = rng.choice(points.shape[0], size=count, replace=False)
            cloud_chunks.append(points[indices])
    local_union = np.concatenate(all_local_points, axis=0)
    cloud = (
        np.concatenate(cloud_chunks, axis=0)
        if cloud_chunks
        else np.empty((0, 5), dtype=np.float64)
    )
    return pareto_frontier(local_union), local_union, cloud, candidate_count


def representative_points(
    frontier: np.ndarray, limits: list[float]
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for limit in limits:
        feasible = frontier[frontier[:, 0] <= limit]
        if feasible.size == 0:
            continue
        point = feasible[np.argmin(feasible[:, 1])]
        rows.append(
            {
                "storage_limit_gib": limit,
                "storage_gib": float(point[0]),
                "recovery_token_layers": float(point[1]),
                "g1": int(point[2]),
                "g2": int(point[3]),
                "l": int(point[4]),
            }
        )
    return rows


def write_csv(path: Path, frontier: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["mean_storage_gib", "mean_recovery_token_layers", "g1", "g2", "l"]
        )
        for storage, compute, g1, g2, split in frontier:
            writer.writerow(
                [f"{storage:.8f}", f"{compute:.8f}", int(g1), int(g2), int(split)]
            )


def write_plot(
    path: Path, frontier: np.ndarray, cloud: np.ndarray, sample_count: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frontier[np.argsort(frontier[:, 0])]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=240)
    if cloud.size:
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            color="#64748b",
            s=3,
            alpha=0.12,
            linewidths=0,
            rasterized=True,
        )
    ax.plot(
        ordered[:, 0],
        ordered[:, 1],
        color="#2563eb",
        linewidth=0.8,
        alpha=0.75,
    )
    ax.scatter(
        ordered[:, 0],
        ordered[:, 1],
        color="#2563eb",
        s=8,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean DRAM checkpoint storage (GiB)")
    ax.set_ylabel("Mean recovery compute (hidden-state token-layers)")
    ax.set_title("DeepSeek V4 Flash g1/g2 checkpoint Pareto frontier")
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.7, alpha=0.75)
    ax.grid(True, which="minor", color="#e2e8f0", linewidth=0.45, alpha=0.55)
    ax.text(
        0.99,
        0.02,
        f"shared N samples={sample_count:,} · Pareto points={len(frontier):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#475569",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


FRAGMENT_TEMPLATE = r"""<div id="dsv4-g1-g2-pareto" class="viz-root">
  <h2 id="dsv4-original-title" class="dsv4-chart-title">双 gap Pareto 前沿（g₁ ≤ g₂）</h2>
  <div class="viz-controls">
    <label class="form-label" for="dsv4-l-input">分割层 l（越大越靠输出侧）：<output id="dsv4-l-out">42</output>
      <input class="form-range" id="dsv4-l-input" type="range" min="1" max="42" value="42" step="1">
    </label>
    <label class="form-label" for="dsv4-g1-input">输出侧 gap g₁：<output id="dsv4-g1-out">1024</output>
      <input class="form-range" id="dsv4-g1-input" type="range" min="1" max="8192" value="1024" step="1">
    </label>
    <label class="form-label" for="dsv4-g2-input">输入侧 gap g₂：<output id="dsv4-g2-out">1024</output>
      <input class="form-range" id="dsv4-g2-input" type="range" min="1" max="8192" value="1024" step="1">
    </label>
  </div>
  <div class="viz-row text-muted text-small" aria-live="polite">
    <span id="dsv4-selected-info"></span>
    <span>· 候选云</span>
    <span>● Pareto 点</span>
    <span>○ 当前选择</span>
    <span id="dsv4-sample-info"></span>
  </div>
  <div class="dsv4-plot" id="dsv4-plot">
    <canvas id="dsv4-canvas" role="img" aria-label="DSV4 Flash 两段 checkpoint gap 的平均存储量与平均恢复计算量散点图和 Pareto 前沿"></canvas>
    <div class="tooltip" id="dsv4-tooltip"></div>
  </div>
</div>
<style>
#dsv4-g1-g2-pareto { width: 100%; color: var(--foreground); }
#dsv4-g1-g2-pareto .dsv4-chart-title { margin-bottom: 12px; }
#dsv4-g1-g2-pareto .dsv4-plot { position: relative; width: 100%; }
#dsv4-g1-g2-pareto canvas { display: block; width: 100%; height: 650px; }
#dsv4-g1-g2-pareto .tooltip { display: none; position: absolute; pointer-events: none; max-width: 280px; }
@media (max-width: 600px) { #dsv4-g1-g2-pareto canvas { height: 480px; } }
</style>
<script>
(() => {
  const root = document.getElementById('dsv4-g1-g2-pareto');
  const DATA = __POINTS_JSON__;
  const META = __META_JSON__;
  const canvas = root.querySelector('#dsv4-canvas');
  const plot = root.querySelector('#dsv4-plot');
  const tip = root.querySelector('#dsv4-tooltip');
  const lInput = root.querySelector('#dsv4-l-input');
  const g1Input = root.querySelector('#dsv4-g1-input');
  const g2Input = root.querySelector('#dsv4-g2-input');
  const lOut = root.querySelector('#dsv4-l-out');
  const g1Out = root.querySelector('#dsv4-g1-out');
  const g2Out = root.querySelector('#dsv4-g2-out');
  const selectedInfo = root.querySelector('#dsv4-selected-info');
  const sampleInfo = root.querySelector('#dsv4-sample-info');
  const ctx = canvas.getContext('2d');
  const colorProbe = document.createElement('span');
  colorProbe.setAttribute('aria-hidden', 'true');
  colorProbe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
  root.appendChild(colorProbe);
  const FRONTIER_INDICES = DATA.map((point, index) => point[5] ? index : -1).filter(index => index >= 0);
  const xLogs = DATA.map(point => Math.log10(point[0]));
  const yLogs = DATA.map(point => Math.log10(point[1]));
  const xMin = Math.min(...xLogs), xMax = Math.max(...xLogs);
  const yMin = Math.min(...yLogs), yMax = Math.max(...yLogs);
  let screen = [], spatial = new Map(), geometry = null, selectedIndex = FRONTIER_INDICES[0], hoverFrame = 0;
  const themeColor = name => {
    colorProbe.style.color = `var(${name})`;
    return getComputedStyle(colorProbe).color;
  };
  const format = (value, digits = 2) => value.toLocaleString('zh-CN', { maximumFractionDigits: digits });
  const ticks = (low, high) => {
    const values = [];
    for (let exponent = Math.floor(low); exponent <= Math.ceil(high); exponent++) for (const multiplier of [1, 2, 5]) {
      const value = multiplier * 10 ** exponent, logValue = Math.log10(value);
      if (logValue >= low && logValue <= high) values.push([value, multiplier === 1]);
    }
    return values;
  };
  const pointText = point => `l=${point[4]}，g₁=${point[2]}，g₂=${point[3]}，${format(point[0], 4)} GiB，${format(point[1], 2)} token-layer${point[5] ? '，Pareto 点' : '，候选点'}`;
  function updateControls(point) {
    lInput.value = String(point[4]); g1Input.value = String(point[2]); g2Input.value = String(point[3]);
    lOut.value = String(point[4]); g1Out.value = String(point[2]); g2Out.value = String(point[3]);
  }
  function closestToControls() {
    const targetL = Number(lInput.value), targetG1 = Number(g1Input.value), targetG2 = Number(g2Input.value);
    let best = Infinity, index = selectedIndex;
    for (let i = 0; i < DATA.length; i++) {
      const point = DATA[i];
      if (point[4] !== targetL) continue;
      const distance = Math.abs(Math.log(point[2] / targetG1)) + Math.abs(Math.log(point[3] / targetG2));
      if (distance < best || (distance === best && point[5] > DATA[index][5])) { best = distance; index = i; }
    }
    return index;
  }
  function render() {
    const width = Math.max(320, plot.clientWidth), height = width < 600 ? 480 : 650, ratio = window.devicePixelRatio || 1;
    canvas.style.height = `${height}px`; canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const margin = { left: width < 600 ? 62 : 82, right: 18, top: 18, bottom: 60 };
    const innerWidth = width - margin.left - margin.right, innerHeight = height - margin.top - margin.bottom;
    const scaleX = value => margin.left + (Math.log10(value) - xMin) / (xMax - xMin) * innerWidth;
    const scaleY = value => margin.top + innerHeight - (Math.log10(value) - yMin) / (yMax - yMin) * innerHeight;
    geometry = { scaleX, scaleY };
    const foreground = themeColor('--foreground'), muted = themeColor('--muted-foreground'), grid = themeColor('--border');
    const cloudColor = muted, paretoColor = themeColor('--viz-series-1'), selectedColor = themeColor('--viz-series-2'), background = themeColor('--background');
    ctx.font = getComputedStyle(root).font; ctx.strokeStyle = grid; ctx.fillStyle = muted; ctx.lineWidth = 1;
    for (const [value, major] of ticks(xMin, xMax)) {
      const x = scaleX(value); ctx.globalAlpha = major ? 0.55 : 0.22; ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + innerHeight); ctx.stroke();
      if (major) { ctx.globalAlpha = 1; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(format(value, value < 10 ? 1 : 0), x, margin.top + innerHeight + 8); }
    }
    for (const [value, major] of ticks(yMin, yMax)) {
      const y = scaleY(value); ctx.globalAlpha = major ? 0.55 : 0.22; ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + innerWidth, y); ctx.stroke();
      if (major) { ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(format(value, 0), margin.left - 8, y); }
    }
    ctx.globalAlpha = 1; ctx.fillStyle = foreground; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('平均 DRAM checkpoint 存储量（GiB，对数）', margin.left + innerWidth / 2, height - 4);
    ctx.save(); ctx.translate(16, margin.top + innerHeight / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('平均恢复计算量（token-layer，对数）', 0, 0); ctx.restore();
    screen = new Array(DATA.length); spatial = new Map(); const cell = 14;
    for (let i = 0; i < DATA.length; i++) {
      const point = DATA[i], x = scaleX(point[0]), y = scaleY(point[1]); screen[i] = [x, y];
      if (!point[5]) { ctx.globalAlpha = 0.16; ctx.fillStyle = cloudColor; ctx.fillRect(x - 0.7, y - 0.7, 1.4, 1.4); }
      const key = `${Math.floor(x / cell)}:${Math.floor(y / cell)}`; if (!spatial.has(key)) spatial.set(key, []); spatial.get(key).push(i);
    }
    ctx.globalAlpha = 0.92; ctx.fillStyle = paretoColor;
    for (const index of FRONTIER_INDICES) { const [x, y] = screen[index]; ctx.beginPath(); ctx.arc(x, y, 1.8, 0, Math.PI * 2); ctx.fill(); }
    const selected = DATA[selectedIndex]; ctx.globalAlpha = 1; ctx.fillStyle = background; ctx.strokeStyle = selectedColor; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(scaleX(selected[0]), scaleY(selected[1]), 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    selectedInfo.textContent = `当前：${pointText(selected)}`; ctx.globalAlpha = 1;
  }
  function nearest(clientX, clientY) {
    if (!geometry) return null;
    const rect = canvas.getBoundingClientRect(), x = clientX - rect.left, y = clientY - rect.top, cell = 14;
    const cellX = Math.floor(x / cell), cellY = Math.floor(y / cell); let best = 144, index = -1;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const candidate of spatial.get(`${cellX + dx}:${cellY + dy}`) || []) {
      const deltaX = screen[candidate][0] - x, deltaY = screen[candidate][1] - y, distance = deltaX * deltaX + deltaY * deltaY;
      if (distance < best || (distance === best && DATA[candidate][5] > (index >= 0 ? DATA[index][5] : -1))) { best = distance; index = candidate; }
    }
    return index < 0 ? null : { index, x, y };
  }
  canvas.addEventListener('pointermove', event => {
    cancelAnimationFrame(hoverFrame); hoverFrame = requestAnimationFrame(() => {
      const hit = nearest(event.clientX, event.clientY); if (!hit) { tip.style.display = 'none'; return; }
      tip.textContent = pointText(DATA[hit.index]); tip.style.display = 'block'; const width = tip.offsetWidth, height = tip.offsetHeight;
      tip.style.left = `${Math.max(0, Math.min(plot.clientWidth - width, hit.x + 12))}px`; tip.style.top = `${Math.max(0, hit.y - height - 10)}px`;
    });
  });
  canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  canvas.addEventListener('click', event => {
    const hit = nearest(event.clientX, event.clientY); if (!hit) return; selectedIndex = hit.index; updateControls(DATA[selectedIndex]); render();
  });
  function onControlInput() {
    if (Number(g2Input.value) < Number(g1Input.value)) g2Input.value = g1Input.value;
    lOut.value = lInput.value; g1Out.value = g1Input.value; g2Out.value = g2Input.value;
    selectedIndex = closestToControls(); render();
  }
  lInput.addEventListener('input', onControlInput); g1Input.addEventListener('input', onControlInput); g2Input.addEventListener('input', onControlInput);
  sampleInfo.textContent = `共同 N 样本 ${META.final_n_samples.toLocaleString('zh-CN')} · Pareto 点 ${META.pareto_points.toLocaleString('zh-CN')}`;
  selectedIndex = FRONTIER_INDICES.reduce((best, index) => Math.abs(Math.log(DATA[index][0] / 32)) < Math.abs(Math.log(DATA[best][0] / 32)) ? index : best, FRONTIER_INDICES[0]);
  updateControls(DATA[selectedIndex]); new ResizeObserver(render).observe(plot); render();
})();
</script>
"""


def write_fragment(
    path: Path,
    frontier: np.ndarray,
    cloud: np.ndarray,
    metadata: dict[str, object],
) -> None:
    ordered = frontier[np.argsort(frontier[:, 0])]
    cloud_points = [
        [
            round(float(storage), 7),
            round(float(compute), 5),
            int(g1),
            int(g2),
            int(split),
            0,
        ]
        for storage, compute, g1, g2, split in cloud
    ]
    frontier_points = [
        [
            round(float(storage), 8),
            round(float(compute), 8),
            int(g1),
            int(g2),
            int(split),
            1,
        ]
        for storage, compute, g1, g2, split in ordered
    ]
    points = cloud_points + frontier_points
    fragment = FRAGMENT_TEMPLATE.replace(
        "__POINTS_JSON__", json.dumps(points, separators=(",", ":"))
    ).replace(
        "__META_JSON__", json.dumps(metadata, separators=(",", ":"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=1_000_000)
    parser.add_argument("--n-max", type=int, default=2_000_000)
    parser.add_argument("--coarse-n-samples", type=int, default=1024)
    parser.add_argument("--refine-n-samples", type=int, default=8192)
    parser.add_argument("--final-n-samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-gap", type=int, default=8192)
    parser.add_argument("--dense-gap-max", type=int, default=256)
    parser.add_argument("--coarse-gap-step", type=int, default=16)
    parser.add_argument("--refine-radius", type=int, default=8)
    parser.add_argument("--refine-stride", type=int, default=1)
    parser.add_argument("--final-radius", type=int, default=3)
    parser.add_argument("--coarse-margin", type=float, default=0.02)
    parser.add_argument("--refine-margin", type=float, default=0.005)
    parser.add_argument("--storage-limit-gib", type=float, default=1024.0)
    parser.add_argument("--g2-block", type=int, default=64)
    parser.add_argument("--cloud-points-per-split", type=int, default=300)
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--output-fragment", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    samples = make_nested_samples(
        args.n_min,
        args.n_max,
        [args.coarse_n_samples, args.refine_n_samples, args.final_n_samples],
        args.seed,
    )
    coarse_gaps = make_gap_grid(
        args.max_gap, args.dense_gap_max, args.coarse_gap_step
    )

    stage_started = perf_counter()
    coarse_evaluator = Evaluator(
        samples[args.coarse_n_samples],
        args.n_min,
        args.n_max,
        args.storage_limit_gib,
        args.g2_block,
        args.workers,
    )
    coarse_frontier, coarse_local, cloud, coarse_candidate_count = evaluate_coarse_grid(
        coarse_evaluator,
        coarse_gaps,
        args.cloud_points_per_split,
        args.seed + 1,
    )
    coarse_seconds = perf_counter() - stage_started

    coarse_seeds = near_frontier(
        coarse_local, coarse_frontier, args.coarse_margin
    )
    refine_candidates = candidate_neighborhood(
        coarse_seeds,
        args.max_gap,
        args.refine_radius,
        args.refine_stride,
        include_full_axes=True,
    )
    stage_started = perf_counter()
    refine_evaluator = Evaluator(
        samples[args.refine_n_samples],
        args.n_min,
        args.n_max,
        args.storage_limit_gib,
        args.g2_block,
        args.workers,
    )
    refine_points = refine_evaluator.evaluate_grouped(
        group_candidates(refine_candidates)
    )
    refine_frontier = pareto_frontier(refine_points)
    refine_seconds = perf_counter() - stage_started

    refine_seeds = near_frontier(
        refine_points, refine_frontier, args.refine_margin
    )
    final_candidates = candidate_neighborhood(
        refine_seeds,
        args.max_gap,
        args.final_radius,
        1,
        include_full_axes=False,
    )
    guard_seed_rows = np.asarray(
        [
            [0.0, 0.0, g1, g2, split]
            for split, g1, g2 in PREVIOUS_REPRESENTATIVE_PARAMETERS
        ],
        dtype=np.float64,
    )
    regression_guard_candidates = candidate_neighborhood(
        guard_seed_rows,
        args.max_gap,
        args.refine_radius,
        1,
        include_full_axes=False,
    )
    final_candidates.update(regression_guard_candidates)
    stage_started = perf_counter()
    final_evaluator = Evaluator(
        samples[args.final_n_samples],
        args.n_min,
        args.n_max,
        args.storage_limit_gib,
        args.g2_block,
        args.workers,
    )
    final_points = final_evaluator.evaluate_grouped(group_candidates(final_candidates))
    frontier = pareto_frontier(final_points)
    final_seconds = perf_counter() - stage_started
    total_seconds = perf_counter() - started

    metadata: dict[str, object] = {
        "model": "DeepSeek-V4-Flash",
        "L": L,
        "W": W,
        "layer_order": "input_to_output",
        "split_definition": "layers_1_through_l_use_g2; layers_l_plus_1_through_L_use_g1",
        "split_depth": "larger_l_is_deeper_and_closer_to_output",
        "g1_side": "near_output",
        "g2_side": "far_from_output",
        "N_min": args.n_min,
        "N_max": args.n_max,
        "N_workload": "uniform_integer_interval",
        "N_sampling": "nested_uniform_without_replacement_with_shared_samples_and_endpoints",
        "random_seed": args.seed,
        "coarse_n_samples": args.coarse_n_samples,
        "refine_n_samples": args.refine_n_samples,
        "final_n_samples": args.final_n_samples,
        "storage_mean": "exact_over_full_N_interval",
        "recovery_mean": "sampled_exact_discrete_formula",
        "metrics": ["mean_storage_gib", "mean_recovery_token_layers"],
        "gap_grid": {
            "max_gap": args.max_gap,
            "dense_integer_range": [1, args.dense_gap_max],
            "coarse_step_above_dense_range": args.coarse_gap_step,
            "layer_hit_boundaries_included": True,
            "grid_size": int(coarse_gaps.size),
        },
        "refinement": {
            "coarse_relative_margin": args.coarse_margin,
            "medium_radius": args.refine_radius,
            "medium_stride": args.refine_stride,
            "medium_full_integer_axes": True,
            "refine_relative_margin": args.refine_margin,
            "final_radius": args.final_radius,
            "final_stride": 1,
        },
        "storage_limit_gib": args.storage_limit_gib,
        "workers": args.workers,
        "coarse_candidate_points_after_storage_filter": coarse_candidate_count,
        "coarse_local_pareto_points": int(coarse_local.shape[0]),
        "coarse_near_frontier_seed_points": int(coarse_seeds.shape[0]),
        "refine_candidate_parameters": len(refine_candidates),
        "refine_points_after_storage_filter": int(refine_points.shape[0]),
        "refine_near_frontier_seed_points": int(refine_seeds.shape[0]),
        "final_candidate_parameters": len(final_candidates),
        "previous_frontier_guard_parameters": len(
            PREVIOUS_REPRESENTATIVE_PARAMETERS
        ),
        "previous_frontier_guard_candidates": len(regression_guard_candidates),
        "final_points_after_storage_filter": int(final_points.shape[0]),
        "pareto_points": int(frontier.shape[0]),
        "interactive_cloud_points": int(cloud.shape[0]),
        "runtime_seconds": {
            "coarse": coarse_seconds,
            "refine": refine_seconds,
            "final": final_seconds,
            "total": total_seconds,
        },
        "representative_points": representative_points(
            frontier, [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ),
    }

    write_csv(args.output_csv, frontier)
    write_plot(args.output_png, frontier, cloud, args.final_n_samples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_fragment is not None:
        write_fragment(args.output_fragment, frontier, cloud, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"PNG: {args.output_png.resolve()}")
    if args.output_fragment is not None:
        print(f"Fragment: {args.output_fragment.resolve()}")


if __name__ == "__main__":
    main()
