#!/usr/bin/env python3
"""Compare the published two-gap Pareto frontier with exhaustive g1 == g2.

The sampled comparison uses the same 32,768 common N values as the published
experiment.  The representative-point validation additionally evaluates every
integer N in [1,000,000, 2,000,000].
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from plot_checkpoint_g1_g2_pareto import (
    D,
    GIB,
    L,
    TOTAL_BYTES,
    exact_mean_checkpoint_counts,
    make_nested_samples,
    pareto_frontier,
    triangle_removed,
)


HERE = Path(__file__).resolve().parent
DEFAULT_UNRESTRICTED_CSV = HERE / "dsv4_checkpoint_g1_g2_pareto.csv"
DEFAULT_UNRESTRICTED_JSON = HERE / "dsv4_checkpoint_g1_g2_metadata.json"
DEFAULT_EQUAL_CSV = HERE / "dsv4_checkpoint_equal_g_pareto.csv"
DEFAULT_EQUAL_JSON = HERE / "dsv4_checkpoint_equal_g_metadata.json"
DEFAULT_HTML = HERE / "dsv4_checkpoint_g1_g2_pareto_interactive.html"

VIS_START = "<!-- equal-gap-comparison:start -->"
VIS_END = "<!-- equal-gap-comparison:end -->"
FLOAT_STORAGE_TOLERANCE_GIB = 5e-8


def equal_gap_compute(phases: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Return C for the optimal equal-gap split l=L-1, vectorized by gap."""
    return (
        (L - 1) * phases
        + gaps
        - triangle_removed(phases, L)
    )


def evaluate_sampled_equal_gaps(
    ns: np.ndarray,
    n_min: int,
    n_max: int,
    max_gap: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaps = np.arange(1, max_gap + 1, dtype=np.int64)
    storage = (
        TOTAL_BYTES * exact_mean_checkpoint_counts(gaps, n_min, n_max) / GIB
    )
    sampled_compute = np.empty(gaps.size, dtype=np.float64)
    for start in range(0, gaps.size, block_size):
        stop = min(start + block_size, gaps.size)
        block = gaps[start:stop]
        phases = ns[:, None] % block[None, :]
        sampled_compute[start:stop] = equal_gap_compute(
            phases, block[None, :]
        ).mean(axis=0)
    return gaps, storage, sampled_compute


def residue_counts(gap: int, n_min: int, n_max: int) -> np.ndarray:
    """Count each N mod gap over the complete inclusive integer interval."""

    def prefix(x: int) -> np.ndarray:
        if x < 0:
            return np.zeros(gap, dtype=np.int64)
        quotient, remainder = divmod(x + 1, gap)
        counts = np.full(gap, quotient, dtype=np.int64)
        counts[:remainder] += 1
        return counts

    return prefix(n_max) - prefix(n_min - 1)


def evaluate_exact_equal_gaps(
    gaps: np.ndarray,
    n_min: int,
    n_max: int,
) -> np.ndarray:
    population = n_max - n_min + 1
    exact_compute = np.empty(gaps.size, dtype=np.float64)
    for index, raw_gap in enumerate(gaps):
        gap = int(raw_gap)
        phases = np.arange(gap, dtype=np.int64)
        counts = residue_counts(gap, n_min, n_max)
        compute = equal_gap_compute(phases, np.asarray(gap, dtype=np.int64))
        exact_compute[index] = float(np.dot(counts, compute)) / population
    return exact_compute


def exact_two_gap_compute(
    ns: np.ndarray,
    split: int,
    g1: int,
    g2: int,
) -> float:
    r1 = ns % g1
    boundary1 = ns - np.minimum(r1, split * D)
    r2 = boundary1 % g2
    compute = (
        split * r1
        - triangle_removed(r1, split)
        + (L - split) * g2
        - triangle_removed(r2, L - split)
    )
    return float(compute.mean())


def verify_equal_gap_reduction(ns: np.ndarray, gaps: list[int]) -> None:
    """Check the closed form and l=L-1 optimum against the two-gap formula."""
    check_ns = ns[: min(ns.size, 2048)]
    for gap in gaps:
        phases = check_ns % gap
        common_removed = triangle_removed(phases, L)
        means: list[float] = []
        for split in range(1, L):
            r2 = np.maximum(phases - split * D, 0)
            direct = (
                split * phases
                - triangle_removed(phases, split)
                + (L - split) * gap
                - triangle_removed(r2, L - split)
            )
            reduced = split * phases + (L - split) * gap - common_removed
            if not np.array_equal(direct, reduced):
                raise AssertionError(f"equal-gap reduction failed for g={gap}, l={split}")
            means.append(float(reduced.mean()))
        if int(np.argmin(means)) + 1 != L - 1:
            raise AssertionError(f"l={L - 1} is not optimal for g={gap}")


def read_frontier(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                [
                    float(row["mean_storage_gib"]),
                    float(row["mean_recovery_token_layers"]),
                    float(row["g1"]),
                    float(row["g2"]),
                    float(row["l"]),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def best_under(frontier: np.ndarray, storage_limit: float) -> np.ndarray:
    ordered = frontier[np.argsort(frontier[:, 0])]
    index = int(
        np.searchsorted(
            ordered[:, 0],
            storage_limit + FLOAT_STORAGE_TOLERANCE_GIB,
            side="right",
        )
        - 1
    )
    if index < 0:
        raise ValueError(f"no feasible point below {storage_limit} GiB")
    return ordered[index]


def build_comparison_curve(
    unrestricted: np.ndarray,
    equal_frontier: np.ndarray,
) -> np.ndarray:
    unrestricted = unrestricted[np.argsort(unrestricted[:, 0])]
    rows: list[list[float]] = []
    for point in unrestricted:
        try:
            equal = best_under(equal_frontier, float(point[0]))
        except ValueError:
            continue
        penalty = (equal[1] / point[1] - 1.0) * 100.0
        rows.append(
            [
                float(point[0]),
                float(penalty),
                float(equal[2]),
                float(point[2]),
                float(point[3]),
                float(point[4]),
                float(equal[1]),
                float(point[1]),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def downsample_extrema(points: np.ndarray, bins: int) -> np.ndarray:
    if points.shape[0] <= bins * 4:
        return points
    logs = np.log10(points[:, 0])
    edges = np.linspace(float(logs.min()), float(logs.max()), bins + 1)
    selected: set[int] = {0, points.shape[0] - 1}
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        indices = np.flatnonzero((logs >= low) & (logs <= high))
        if indices.size:
            selected.update(
                (
                    int(indices[0]),
                    int(indices[-1]),
                    int(indices[np.argmin(points[indices, 1])]),
                    int(indices[np.argmax(points[indices, 1])]),
                )
            )
    return points[sorted(selected)]


def write_equal_frontier_csv(
    path: Path,
    frontier: np.ndarray,
    exact_compute: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "mean_storage_gib",
                "sampled_mean_recovery_token_layers",
                "exact_mean_recovery_token_layers",
                "g",
                "l",
            ]
        )
        for storage, sampled, g1, _, split in frontier:
            gap = int(g1)
            writer.writerow(
                [
                    f"{storage:.8f}",
                    f"{sampled:.8f}",
                    f"{exact_compute[gap - 1]:.8f}",
                    gap,
                    int(split),
                ]
            )


def build_representative_comparison(
    published_metadata: dict[str, object],
    unrestricted_frontier: np.ndarray,
    equal_frontier: np.ndarray,
    gaps: np.ndarray,
    storage: np.ndarray,
    exact_equal: np.ndarray,
    full_ns: np.ndarray,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    representatives = published_metadata["representative_points"]
    assert isinstance(representatives, list)
    for raw in representatives:
        assert isinstance(raw, dict)
        storage_cap = float(raw["storage_gib"])
        published = best_under(unrestricted_frontier, storage_cap)
        sampled_equal = best_under(equal_frontier, storage_cap)
        feasible = storage <= storage_cap + FLOAT_STORAGE_TOLERANCE_GIB
        feasible_indices = np.flatnonzero(feasible)
        exact_index = int(feasible_indices[np.argmin(exact_equal[feasible])])
        split = int(raw["l"])
        g1 = int(raw["g1"])
        g2 = int(raw["g2"])
        dual_exact = exact_two_gap_compute(full_ns, split, g1, g2)
        equal_exact = float(exact_equal[exact_index])
        rows.append(
            {
                "nominal_storage_limit_gib": float(raw["storage_limit_gib"]),
                "comparison_storage_cap_gib": storage_cap,
                "dual_storage_gib": float(raw["storage_gib"]),
                "dual_sampled_recovery": float(raw["recovery_token_layers"]),
                "dual_exact_recovery": dual_exact,
                "dual_g1": g1,
                "dual_g2": g2,
                "dual_l": split,
                "equal_sampled_g": int(sampled_equal[2]),
                "equal_sampled_storage_gib": float(sampled_equal[0]),
                "equal_sampled_recovery": float(sampled_equal[1]),
                "sampled_penalty_percent": float(
                    (sampled_equal[1] / published[1] - 1.0) * 100.0
                ),
                "equal_exact_g": int(gaps[exact_index]),
                "equal_exact_storage_gib": float(storage[exact_index]),
                "equal_exact_recovery": equal_exact,
                "exact_penalty_percent": float(
                    (equal_exact / dual_exact - 1.0) * 100.0
                ),
            }
        )
    return rows


def comparison_fragment(curve: np.ndarray, representatives: list[dict[str, int | float]]) -> str:
    curve_rows = [
        [
            round(float(row[0]), 8),
            round(float(row[1]), 6),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            round(float(row[6]), 6),
            round(float(row[7]), 6),
        ]
        for row in curve
    ]
    representative_rows = [
        [
            round(float(row["comparison_storage_cap_gib"]), 8),
            round(float(row["sampled_penalty_percent"]), 6),
            int(row["equal_sampled_g"]),
            int(row["dual_g1"]),
            int(row["dual_g2"]),
            int(row["dual_l"]),
            round(float(row["equal_sampled_recovery"]), 6),
            round(float(row["dual_sampled_recovery"]), 6),
            float(row["nominal_storage_limit_gib"]),
        ]
        for row in representatives
    ]
    curve_json = json.dumps(curve_rows, separators=(",", ":"))
    representatives_json = json.dumps(representative_rows, separators=(",", ":"))
    return f"""{VIS_START}
<div id="dsv4-equal-gap-comparison" class="viz-root">
  <h2 class="dsv4-chart-title">同 gap 对照：相对恢复量差异（g₁ = g₂）</h2>
  <div class="viz-row text-muted text-small" aria-live="polite">
    <span>同一存储上限下，强制 g₁=g₂ 相对双 g 前沿</span>
    <span>● 代表预算</span>
    <span id="dsv4-equal-selected"></span>
  </div>
  <div class="dsv4-equal-plot" id="dsv4-equal-plot">
    <canvas id="dsv4-equal-canvas" role="img" aria-label="强制 g1 等于 g2 时，相对双 gap Pareto 前沿的平均恢复计算量百分比差异；负值表示同 g 更好"></canvas>
    <div class="tooltip" id="dsv4-equal-tooltip"></div>
  </div>
</div>
<style>
#dsv4-equal-gap-comparison {{ width: 100%; margin-top: 36px; color: var(--foreground); }}
#dsv4-g1-g2-pareto .dsv4-chart-title,
#dsv4-equal-gap-comparison .dsv4-chart-title {{ margin-bottom: 12px; }}
#dsv4-equal-gap-comparison .dsv4-equal-plot {{ position: relative; width: 100%; }}
#dsv4-equal-gap-comparison canvas {{ display: block; width: 100%; height: 430px; }}
#dsv4-equal-gap-comparison .tooltip {{ display: none; position: absolute; pointer-events: none; max-width: 300px; }}
@media (max-width: 600px) {{ #dsv4-equal-gap-comparison canvas {{ height: 360px; }} }}
</style>
<script>
(() => {{
  const root = document.getElementById('dsv4-equal-gap-comparison');
  const CURVE = {curve_json};
  const REPS = {representatives_json};
  const canvas = root.querySelector('#dsv4-equal-canvas');
  const plot = root.querySelector('#dsv4-equal-plot');
  const tip = root.querySelector('#dsv4-equal-tooltip');
  const selectedInfo = root.querySelector('#dsv4-equal-selected');
  const ctx = canvas.getContext('2d');
  const colorProbe = document.createElement('span');
  colorProbe.setAttribute('aria-hidden', 'true');
  colorProbe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
  root.appendChild(colorProbe);
  const color = name => {{ colorProbe.style.color = `var(${{name}})`; return getComputedStyle(colorProbe).color; }};
  const format = (value, digits = 2) => value.toLocaleString('zh-CN', {{ maximumFractionDigits: digits }});
  const xLogs = CURVE.map(point => Math.log10(point[0]));
  const xMin = Math.min(...xLogs), xMax = Math.max(...xLogs);
  const rawY = CURVE.map(point => point[1]).concat(REPS.map(point => point[1]), [0]);
  const rawMin = Math.min(...rawY), rawMax = Math.max(...rawY);
  const rawSpan = Math.max(rawMax - rawMin, 0.02);
  const yMin = rawMin - rawSpan * 0.1, yMax = rawMax + rawSpan * 0.1;
  let screen = [], repScreen = [], geometry = null, selected = Math.min(3, REPS.length - 1), hoverFrame = 0;
  const pointText = point => `存储≤${{format(point[0], 4)}} GiB：同 g=${{point[2]}}，双 g=${{point[3]}}/${{point[4]}}（l=${{point[5]}}），恢复量差异 ${{point[1] >= 0 ? '+' : ''}}${{format(point[1], 3)}}%`;
  const niceStep = span => {{
    const raw = span / 5, power = 10 ** Math.floor(Math.log10(raw)), scaled = raw / power;
    return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
  }};
  function render() {{
    const width = Math.max(320, plot.clientWidth), height = width < 600 ? 360 : 430, ratio = window.devicePixelRatio || 1;
    canvas.style.height = `${{height}}px`; canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const margin = {{ left: width < 600 ? 62 : 82, right: 18, top: 18, bottom: 60 }};
    const innerWidth = width - margin.left - margin.right, innerHeight = height - margin.top - margin.bottom;
    const scaleX = value => margin.left + (Math.log10(value) - xMin) / (xMax - xMin) * innerWidth;
    const scaleY = value => margin.top + innerHeight - (value - yMin) / (yMax - yMin) * innerHeight;
    geometry = {{ scaleX, scaleY }};
    const foreground = color('--foreground'), muted = color('--muted-foreground'), grid = color('--border');
    const lineColor = color('--viz-series-1'), repColor = color('--viz-series-2'), background = color('--background');
    ctx.font = getComputedStyle(root).font; ctx.fillStyle = muted; ctx.strokeStyle = grid; ctx.lineWidth = 1;
    for (let exponent = Math.floor(xMin); exponent <= Math.ceil(xMax); exponent++) for (const multiplier of [1, 2, 5]) {{
      const value = multiplier * 10 ** exponent, logValue = Math.log10(value); if (logValue < xMin || logValue > xMax) continue;
      const x = scaleX(value), major = multiplier === 1; ctx.globalAlpha = major ? 0.55 : 0.22; ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + innerHeight); ctx.stroke();
      if (major) {{ ctx.globalAlpha = 1; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(format(value, value < 10 ? 1 : 0), x, margin.top + innerHeight + 8); }}
    }}
    const step = niceStep(yMax - yMin), firstTick = Math.ceil(yMin / step) * step;
    for (let value = firstTick; value <= yMax + step * 0.1; value += step) {{
      const y = scaleY(value); ctx.globalAlpha = Math.abs(value) < step * 1e-6 ? 0.9 : 0.35; ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + innerWidth, y); ctx.stroke();
      ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(`${{format(value, 2)}}%`, margin.left - 8, y);
    }}
    ctx.globalAlpha = 1; ctx.fillStyle = foreground; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('平均 DRAM checkpoint 存储上限（GiB，对数）', margin.left + innerWidth / 2, height - 4);
    ctx.save(); ctx.translate(16, margin.top + innerHeight / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('同 g 恢复量差异（%，负值更好）', 0, 0); ctx.restore();
    screen = CURVE.map(point => [scaleX(point[0]), scaleY(point[1])]);
    ctx.strokeStyle = lineColor; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.9; ctx.beginPath();
    screen.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
    repScreen = REPS.map(point => [scaleX(point[0]), scaleY(point[1])]); ctx.fillStyle = repColor; ctx.globalAlpha = 1;
    repScreen.forEach(([x, y]) => {{ ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill(); }});
    if (selected >= 0) {{ const [x, y] = repScreen[selected]; ctx.fillStyle = background; ctx.strokeStyle = repColor; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, 6.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); selectedInfo.textContent = `· 当前：${{pointText(REPS[selected])}}`; }}
  }}
  function nearest(clientX, clientY) {{
    if (!geometry) return null; const rect = canvas.getBoundingClientRect(), x = clientX - rect.left, y = clientY - rect.top;
    let best = 144, result = null;
    repScreen.forEach((point, index) => {{ const dx = point[0] - x, dy = point[1] - y, distance = dx * dx + dy * dy; if (distance < best) {{ best = distance; result = {{ kind: 'rep', index, x, y }}; }} }});
    screen.forEach((point, index) => {{ const dx = point[0] - x, dy = point[1] - y, distance = dx * dx + dy * dy; if (distance < best) {{ best = distance; result = {{ kind: 'curve', index, x, y }}; }} }});
    return result;
  }}
  canvas.addEventListener('pointermove', event => {{ cancelAnimationFrame(hoverFrame); hoverFrame = requestAnimationFrame(() => {{
    const hit = nearest(event.clientX, event.clientY); if (!hit) {{ tip.style.display = 'none'; return; }} const point = hit.kind === 'rep' ? REPS[hit.index] : CURVE[hit.index];
    tip.textContent = pointText(point); tip.style.display = 'block'; const tipWidth = tip.offsetWidth, tipHeight = tip.offsetHeight;
    tip.style.left = `${{Math.max(0, Math.min(plot.clientWidth - tipWidth, hit.x + 12))}}px`; tip.style.top = `${{Math.max(0, hit.y - tipHeight - 10)}}px`;
  }}); }});
  canvas.addEventListener('pointerleave', () => {{ tip.style.display = 'none'; }});
  canvas.addEventListener('click', event => {{ const hit = nearest(event.clientX, event.clientY); if (!hit || hit.kind !== 'rep') return; selected = hit.index; render(); }});
  new ResizeObserver(render).observe(plot); render();
}})();
</script>
{VIS_END}"""


def update_standalone_html(path: Path, fragment: str) -> None:
    outer = path.read_text(encoding="utf-8")
    match = re.search(r'(srcdoc=")(.*?)("></iframe>)', outer, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"could not find iframe srcdoc in {path}")
    inner = html.unescape(match.group(2))
    original_root = '<div id="dsv4-g1-g2-pareto" class="viz-root">'
    original_title = (
        '<h2 id="dsv4-original-title" class="dsv4-chart-title">'
        '双 gap Pareto 前沿（g₁ ≤ g₂）</h2>'
    )
    if 'id="dsv4-original-title"' in inner:
        inner, title_count = re.subn(
            r'<h2 id="dsv4-original-title"[^>]*>.*?</h2>',
            original_title,
            inner,
            count=1,
            flags=re.DOTALL,
        )
        if title_count != 1:
            raise ValueError("could not update the original visualization title")
    elif original_root in inner:
        inner = inner.replace(
            original_root,
            original_root + "\n  " + original_title,
            1,
        )
    else:
        raise ValueError("could not find the original visualization root")
    if VIS_START in inner:
        pattern = re.escape(VIS_START) + r".*?" + re.escape(VIS_END)
        inner, count = re.subn(pattern, fragment, inner, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError("could not replace the existing equal-gap visualization")
    else:
        anchor = '<script src="https://unpkg.com/@floating-ui/core'
        position = inner.find(anchor)
        if position < 0:
            position = inner.rfind("</body>")
        if position < 0:
            raise ValueError("could not find an insertion point in the visualization")
        inner = inner[:position] + fragment + "\n\n" + inner[position:]
    encoded = html.escape(inner, quote=True)
    updated = outer[: match.start(2)] + encoded + outer[match.end(2) :]
    path.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=1_000_000)
    parser.add_argument("--n-max", type=int, default=2_000_000)
    parser.add_argument("--final-n-samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-gap", type=int, default=8192)
    parser.add_argument("--storage-limit-gib", type=float, default=1024.0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--curve-bins", type=int, default=600)
    parser.add_argument("--unrestricted-csv", type=Path, default=DEFAULT_UNRESTRICTED_CSV)
    parser.add_argument("--unrestricted-json", type=Path, default=DEFAULT_UNRESTRICTED_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_EQUAL_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_EQUAL_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    published_metadata = json.loads(args.unrestricted_json.read_text(encoding="utf-8"))
    samples = make_nested_samples(
        args.n_min,
        args.n_max,
        [args.final_n_samples],
        args.seed,
    )[args.final_n_samples]
    verification_gaps = [1, 21, 43, 85, 169, 341, 678, 1351, 2747, 6118, 8192]
    verify_equal_gap_reduction(samples, verification_gaps)

    stage = perf_counter()
    gaps, storage, sampled_compute = evaluate_sampled_equal_gaps(
        samples,
        args.n_min,
        args.n_max,
        args.max_gap,
        args.block_size,
    )
    sampled_seconds = perf_counter() - stage
    all_equal_points = np.column_stack(
        (
            storage,
            sampled_compute,
            gaps,
            gaps,
            np.full(gaps.size, L - 1),
        )
    )
    all_equal_points = all_equal_points[all_equal_points[:, 0] <= args.storage_limit_gib]
    equal_frontier = pareto_frontier(all_equal_points)

    stage = perf_counter()
    exact_equal = evaluate_exact_equal_gaps(gaps, args.n_min, args.n_max)
    exact_seconds = perf_counter() - stage
    unrestricted_frontier = read_frontier(args.unrestricted_csv)
    full_ns = np.arange(args.n_min, args.n_max + 1, dtype=np.int64)
    representatives = build_representative_comparison(
        published_metadata,
        unrestricted_frontier,
        equal_frontier,
        gaps,
        storage,
        exact_equal,
        full_ns,
    )
    curve = build_comparison_curve(unrestricted_frontier, equal_frontier)
    plotted_curve = downsample_extrema(curve, args.curve_bins)
    augmented_frontier = pareto_frontier(
        np.concatenate((unrestricted_frontier, equal_frontier), axis=0)
    )
    equal_rows_on_augmented = int(np.count_nonzero(augmented_frontier[:, 2] == augmented_frontier[:, 3]))
    exact_penalties = np.asarray(
        [float(row["exact_penalty_percent"]) for row in representatives]
    )
    sampled_penalties = curve[:, 1]
    total_seconds = perf_counter() - started
    metadata: dict[str, object] = {
        "model": published_metadata["model"],
        "N_min": args.n_min,
        "N_max": args.n_max,
        "random_seed": args.seed,
        "sampled_recovery_common_N": args.final_n_samples,
        "exact_validation_N": args.n_max - args.n_min + 1,
        "max_gap": args.max_gap,
        "equal_gap_candidate_count": int(all_equal_points.shape[0]),
        "equal_gap_sampled_pareto_points": int(equal_frontier.shape[0]),
        "equal_gap_optimal_split": L - 1,
        "equal_gap_optimal_split_reason": "C(l,g,g)=l*mean(N mod g)+(L-l)*g-sum_k mean((N mod g-kd)_+), so it decreases with l",
        "verified_reduction_gaps": verification_gaps,
        "comparison_curve_points": int(curve.shape[0]),
        "comparison_plot_points": int(plotted_curve.shape[0]),
        "sampled_penalty_percent": {
            "min": float(sampled_penalties.min()),
            "median": float(np.median(sampled_penalties)),
            "p95": float(np.percentile(sampled_penalties, 95)),
            "max": float(sampled_penalties.max()),
            "equal_better_point_count": int(np.count_nonzero(sampled_penalties < -1e-9)),
            "equal_match_point_count": int(np.count_nonzero(np.abs(sampled_penalties) <= 1e-9)),
        },
        "representative_exact_penalty_percent": {
            "min": float(exact_penalties.min()),
            "median": float(np.median(exact_penalties)),
            "max": float(exact_penalties.max()),
            "equal_better_point_count": int(np.count_nonzero(exact_penalties < -1e-9)),
            "equal_match_point_count": int(np.count_nonzero(np.abs(exact_penalties) <= 1e-9)),
        },
        "augmented_frontier_points": int(augmented_frontier.shape[0]),
        "equal_gap_rows_on_augmented_frontier": equal_rows_on_augmented,
        "runtime_seconds": {
            "sampled_equal_gap_exhaustive": sampled_seconds,
            "exact_equal_gap_exhaustive": exact_seconds,
            "total": total_seconds,
        },
        "representative_points": representatives,
    }

    write_equal_frontier_csv(args.output_csv, equal_frontier, exact_equal)
    args.output_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    update_standalone_html(
        args.output_html,
        comparison_fragment(plotted_curve, representatives),
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"JSON: {args.output_json.resolve()}")
    print(f"HTML: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
