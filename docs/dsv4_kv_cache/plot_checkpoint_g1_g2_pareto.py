#!/usr/bin/env python3
"""Sample the DSV4 Flash checkpoint trade-off for independent g1/g2 gaps.

This is a separate experiment from ``plot_checkpoint_pareto.py``.  Layers
1..l use gap g1; layers l+1..L use gap g2, with g2 >= g1.  The program samples
N in [1_000_000, 2_000_000], averages storage and recovery compute over those
N values, rejects candidates above a storage ceiling, and writes only the
global Pareto frontier.

The recovery geometry generalized from the earlier g/2g experiment is:

* r1 = N mod g1
* c1 = N - r1 is the shallow checkpoint boundary
* r2 = c1 mod g2 is its phase in the deep checkpoint grid
* C_low  = l*r1 - r1^2/(2*(W-1))
* C_high = (L-l)*g2 - r2^2/(2*(W-1))

The two complete-triangle constraints are g1-1 <= l*(W-1) and
g2-1 <= (L-l)*(W-1).  Storage uses the exact sampled checkpoint counts
ceil(N/g1) and ceil(N/g2).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


L = 43
W = 128
D = W - 1
GIB = 1024**3

LAYER_CHECKPOINT_BYTES = [74_880, 74_880]
for _ in range(20):
    LAYER_CHECKPOINT_BYTES.extend([157_824, 600_192])
LAYER_CHECKPOINT_BYTES.append(157_824)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "dsv4_checkpoint_g1_g2_pareto.csv"
DEFAULT_JSON = HERE / "dsv4_checkpoint_g1_g2_metadata.json"
DEFAULT_PNG = HERE / "imgs" / "dsv4-checkpoint-g1-g2-pareto.png"
DEFAULT_HTML = HERE / "dsv4_checkpoint_g1_g2_pareto_interactive.html"


def make_n_samples(n_min: int, n_max: int, count: int, seed: int) -> np.ndarray:
    if n_min > n_max:
        raise ValueError("n_min must not exceed n_max")
    if count < 2:
        raise ValueError("n_samples must be at least 2")
    count = min(count, n_max - n_min + 1)
    if count == n_max - n_min + 1:
        return np.arange(n_min, n_max + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    interior = rng.choice(
        n_max - n_min - 1,
        size=count - 2,
        replace=False,
    ).astype(np.int64) + n_min + 1
    return np.sort(np.concatenate(([n_min, n_max], interior)))


def make_gap_grid(max_gap: int, dense_max: int, coarse_step: int) -> np.ndarray:
    if coarse_step < 1:
        raise ValueError("gap_step must be positive")
    dense_max = min(max_gap, max(1, dense_max))
    dense = np.arange(1, dense_max + 1, dtype=np.int64)
    coarse_start = dense_max + coarse_step
    coarse = np.arange(coarse_start, max_gap + 1, coarse_step, dtype=np.int64)

    # Always include layer-boundary feasibility limits, even when they do not
    # lie on the coarse grid.
    boundaries = []
    for l in range(1, L):
        boundaries.extend((l * D + 1, (L - l) * D + 1))
    return np.unique(
        np.concatenate(
            [dense, coarse, np.asarray(boundaries, dtype=np.int64), [max_gap]]
        )
    )


def precompute_basic_means(
    ns: np.ndarray, gaps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cheap per-gap statistics shared by every split and gap pair."""
    gap_count = gaps.size
    mean_ceil = np.empty(gap_count, dtype=np.float64)
    mean_r1 = np.empty(gap_count, dtype=np.float64)
    mean_r1_sq = np.empty(gap_count, dtype=np.float64)
    for i, g1 in enumerate(gaps):
        r1 = ns % g1
        mean_ceil[i] = np.mean((ns + g1 - 1) // g1)
        mean_r1[i] = np.mean(r1)
        mean_r1_sq[i] = np.mean(r1.astype(np.float64) ** 2)
    return mean_ceil, mean_r1, mean_r1_sq


def build_needed_pair_mask(
    gaps: np.ndarray,
    mean_ceil: np.ndarray,
    storage_limit_gib: float,
    prefix_bytes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Reject pairs above the storage limit for every possible split l."""
    needed = np.zeros((gaps.size, gaps.size), dtype=bool)
    feasible_before_storage = 0
    total_bytes = int(prefix_bytes[-1])
    for split in range(1, L):
        a = int(np.searchsorted(gaps, split * D + 1, side="right"))
        b = int(np.searchsorted(gaps, (L - split) * D + 1, side="right"))
        g1v = gaps[:a]
        g2v = gaps[:b]
        pair_valid = g2v[None, :] >= g1v[:, None]
        feasible_before_storage += int(np.count_nonzero(pair_valid))
        low_bytes = int(prefix_bytes[split])
        high_bytes = total_bytes - low_bytes
        storage = (
            low_bytes * mean_ceil[:a, None]
            + high_bytes * mean_ceil[None, :b]
        ) / GIB
        needed[:a, :b] |= pair_valid & (storage <= storage_limit_gib)
    return needed, feasible_before_storage


def precompute_deep_phase_means(
    ns: np.ndarray,
    gaps: np.ndarray,
    needed_pairs: np.ndarray,
    g2_block: int,
    workers: int,
) -> np.ndarray:
    """Compute sampled E[r2^2] only for pairs surviving cheap prefilters."""
    mean_r2_sq = np.full((gaps.size, gaps.size), np.nan, dtype=np.float64)

    def compute_row(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        indices = np.flatnonzero(needed_pairs[i])
        if indices.size == 0:
            return i, indices, np.empty(0, dtype=np.float64)
        g1 = gaps[i]
        c1 = ns - ns % g1
        values = np.empty(indices.size, dtype=np.float64)
        for start in range(0, indices.size, g2_block):
            stop = min(indices.size, start + g2_block)
            block_indices = indices[start:stop]
            phase = c1[:, None] % gaps[block_indices][None, :]
            values[start:stop] = np.mean(
                phase.astype(np.float64) ** 2, axis=0
            )
        return i, indices, values

    row_indices = [i for i in range(gaps.size) if np.any(needed_pairs[i])]
    if workers <= 1:
        for i in row_indices:
            row, indices, values = compute_row(i)
            mean_r2_sq[row, indices] = values
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row, indices, values in executor.map(compute_row, row_indices):
                mean_r2_sq[row, indices] = values
    return mean_r2_sq


def local_frontier(
    storage: np.ndarray,
    compute: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    l: np.ndarray,
) -> np.ndarray:
    """Return indices of nondominated points, keeping one representative tie."""
    if storage.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((compute, storage))
    ordered_compute = compute[order]
    keep = np.zeros(order.size, dtype=bool)
    best = np.inf
    for pos, value in enumerate(ordered_compute):
        if value < best - 1e-10:
            keep[pos] = True
            best = value
    return order[keep]


def evaluate(
    ns: np.ndarray,
    gaps: np.ndarray,
    storage_limit_gib: float,
    g2_block: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    prefix = np.concatenate(([0], np.cumsum(LAYER_CHECKPOINT_BYTES, dtype=np.int64)))
    total_bytes = int(prefix[-1])
    mean_ceil, mean_r1, mean_r1_sq = precompute_basic_means(ns, gaps)
    needed_pairs, feasible_before_storage = build_needed_pair_mask(
        gaps, mean_ceil, storage_limit_gib, prefix
    )
    mean_r2_sq = precompute_deep_phase_means(
        ns, gaps, needed_pairs, g2_block, workers
    )

    frontier_chunks: list[np.ndarray] = []
    candidates_under_limit = 0

    for split in range(1, L):
        low_max = split * D + 1
        high_max = (L - split) * D + 1
        a = int(np.searchsorted(gaps, low_max, side="right"))
        b = int(np.searchsorted(gaps, high_max, side="right"))
        if a == 0 or b == 0:
            continue

        g1v = gaps[:a]
        g2v = gaps[:b]
        pair_valid = g2v[None, :] >= g1v[:, None]

        low_bytes = int(prefix[split])
        high_bytes = total_bytes - low_bytes
        storage = (
            low_bytes * mean_ceil[:a, None]
            + high_bytes * mean_ceil[None, :b]
        ) / GIB
        valid = pair_valid & (storage <= storage_limit_gib)
        candidates_under_limit += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue

        low_compute = (
            split * mean_r1[:a] - mean_r1_sq[:a] / (2 * D)
        )[:, None]
        compute = (
            low_compute
            + (L - split) * g2v[None, :]
            - mean_r2_sq[:a, :b] / (2 * D)
        )

        ii, jj = np.nonzero(valid)
        s = storage[ii, jj]
        c = compute[ii, jj]
        g1_flat = g1v[ii]
        g2_flat = g2v[jj]
        l_flat = np.full(ii.size, split, dtype=np.int64)
        keep = local_frontier(s, c, g1_flat, g2_flat, l_flat)
        frontier_chunks.append(
            np.column_stack(
                (s[keep], c[keep], g1_flat[keep], g2_flat[keep], l_flat[keep])
            )
        )

    merged = np.concatenate(frontier_chunks, axis=0)
    keep = local_frontier(
        merged[:, 0], merged[:, 1], merged[:, 2], merged[:, 3], merged[:, 4]
    )
    frontier = merged[keep]
    frontier = frontier[np.argsort(frontier[:, 0])]

    stats: dict[str, int | float] = {
        "gap_grid_size": int(gaps.size),
        "unique_gap_pairs_after_storage_prefilter": int(
            np.count_nonzero(needed_pairs)
        ),
        "feasible_parameter_points_before_storage_filter": feasible_before_storage,
        "candidate_points_under_storage_limit": candidates_under_limit,
        "pareto_points": int(frontier.shape[0]),
    }
    return frontier, stats


def write_csv(path: Path, frontier: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["storage_gib", "recovery_token_layers", "g1", "g2", "l"]
        )
        for storage, compute, g1, g2, split in frontier:
            writer.writerow(
                [f"{storage:.8f}", f"{compute:.8f}", int(g1), int(g2), int(split)]
            )


def write_plot(path: Path, frontier: np.ndarray, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=240)
    scatter = ax.scatter(
        frontier[:, 0],
        frontier[:, 1],
        c=frontier[:, 4],
        s=10,
        cmap="viridis",
        alpha=0.9,
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean DRAM checkpoint storage (GiB)")
    ax.set_ylabel("Mean recovery compute (hidden-state token-layers)")
    ax.set_title("DeepSeek V4 Flash g1/g2 checkpoint Pareto frontier")
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.7, alpha=0.75)
    ax.grid(True, which="minor", color="#e2e8f0", linewidth=0.45, alpha=0.55)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("split layer l")
    ax.text(
        0.99,
        0.02,
        f"N samples={n_samples:,} · Pareto points={len(frontier):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#475569",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


INTERACTIVE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSV4 Flash g1/g2 checkpoint Pareto</title>
<style>
:root { color-scheme:light dark; --bg:#fff; --fg:#172033; --muted:#64748b; --grid:#cbd5e1; --point:#2563eb; --selected:#dc2626; --panel:#f8fafc; --border:#cbd5e1; }
@media (prefers-color-scheme:dark) { :root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --grid:#334155; --point:#60a5fa; --selected:#fb7185; --panel:#1e293b; --border:#475569; } }
* { box-sizing:border-box; }
body { margin:0; padding:20px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1280px; margin:0 auto; }
h1 { margin:0 0 12px; font-size:20px; font-weight:600; }
.controls { display:flex; align-items:center; flex-wrap:wrap; gap:18px; margin-bottom:8px; }
label { display:flex; align-items:center; gap:8px; }
select { font:inherit; color:var(--fg); background:var(--panel); border:1px solid var(--border); border-radius:5px; padding:5px 8px; }
.status { display:flex; flex-wrap:wrap; gap:14px; margin:8px 0; color:var(--muted); }
.plot { position:relative; width:100%; }
canvas { display:block; width:100%; height:650px; }
.tooltip { display:none; position:absolute; pointer-events:none; max-width:270px; padding:8px 10px; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; box-shadow:0 4px 18px rgb(0 0 0 / 18%); }
.point { color:var(--point); font-weight:700; }
.selected { color:var(--selected); font-weight:700; }
@media (max-width:600px) { body { padding:10px; } canvas { height:480px; } }
</style>
</head>
<body>
<main>
  <h1>DeepSeek V4 Flash g₁/g₂ checkpoint Pareto 前沿</h1>
  <div class="controls">
    <label for="layer-filter">高亮分层点 l
      <select id="layer-filter"><option value="all">全部</option></select>
    </label>
  </div>
  <div class="status" aria-live="polite">
    <span id="selected-info">悬停或点击点查看参数</span>
    <span><span class="point">●</span> Pareto 点</span>
    <span><span class="selected">●</span> 已选择</span>
    <span id="sample-info"></span>
  </div>
  <div class="plot" id="plot">
    <canvas id="canvas" aria-label="DSV4 Flash g1 g2 checkpoint 存储量与恢复计算量 Pareto 散点图"></canvas>
    <div class="tooltip" id="tooltip"></div>
  </div>
</main>
<script>
const DATA=__POINTS_JSON__;
const META=__META_JSON__;
const canvas=document.getElementById('canvas'),plot=document.getElementById('plot'),tip=document.getElementById('tooltip'),layerFilter=document.getElementById('layer-filter'),selectedInfo=document.getElementById('selected-info'),sampleInfo=document.getElementById('sample-info'),ctx=canvas.getContext('2d');
for(let l=1;l<=42;l++){const option=document.createElement('option');option.value=String(l);option.textContent=String(l);layerFilter.appendChild(option);}
sampleInfo.textContent=`N 样本 ${META.N_samples.toLocaleString('zh-CN')} · Pareto 点 ${DATA.length.toLocaleString('zh-CN')}`;
const xMin=Math.log10(Math.min(...DATA.map(p=>p[0]))),xMax=Math.log10(Math.max(...DATA.map(p=>p[0]))),yMin=Math.log10(Math.min(...DATA.map(p=>p[1]))),yMax=Math.log10(Math.max(...DATA.map(p=>p[1])));
let screen=[],spatial=new Map(),geometry=null,selectedIndex=-1,hoverFrame=0;
function color(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
function fmt(v,d=2){return v.toLocaleString('zh-CN',{maximumFractionDigits:d});}
function ticks(lo,hi){const out=[];for(let e=Math.floor(lo);e<=Math.ceil(hi);e++)for(const m of [1,2,5]){const v=m*10**e,lv=Math.log10(v);if(lv>=lo&&lv<=hi)out.push([v,m===1]);}return out;}
function pointText(p){return `g₁=${p[2]}，g₂=${p[3]}，l=${p[4]}；${fmt(p[0],4)} GiB；${fmt(p[1],2)} token-layer`;}
function render(){
  const width=Math.max(320,plot.clientWidth),height=width<600?480:650,ratio=devicePixelRatio||1;canvas.style.height=`${height}px`;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  const margin={left:width<600?58:78,right:18,top:18,bottom:58},iw=width-margin.left-margin.right,ih=height-margin.top-margin.bottom,sx=v=>margin.left+(Math.log10(v)-xMin)/(xMax-xMin)*iw,sy=v=>margin.top+ih-(Math.log10(v)-yMin)/(yMax-yMin)*ih;geometry={sx,sy,width,height,margin,iw,ih};
  ctx.font=`${getComputedStyle(document.body).fontSize} ${getComputedStyle(document.body).fontFamily}`;ctx.strokeStyle=color('--grid');ctx.fillStyle=color('--muted');ctx.lineWidth=1;
  for(const [v,major] of ticks(xMin,xMax)){const x=sx(v);ctx.globalAlpha=major?0.55:0.22;ctx.beginPath();ctx.moveTo(x,margin.top);ctx.lineTo(x,margin.top+ih);ctx.stroke();if(major){ctx.globalAlpha=1;ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(fmt(v,v<10?1:0),x,margin.top+ih+8);}}
  for(const [v,major] of ticks(yMin,yMax)){const y=sy(v);ctx.globalAlpha=major?0.55:0.22;ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(margin.left+iw,y);ctx.stroke();if(major){ctx.globalAlpha=1;ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(fmt(v,0),margin.left-8,y);}}
  ctx.globalAlpha=1;ctx.fillStyle=color('--fg');ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText('平均 DRAM checkpoint 存储量（GiB，对数）',margin.left+iw/2,height-4);ctx.save();ctx.translate(15,margin.top+ih/2);ctx.rotate(-Math.PI/2);ctx.fillText('平均恢复计算量（token-layer，对数）',0,0);ctx.restore();
  screen=new Array(DATA.length);spatial=new Map();const cell=14,filter=layerFilter.value;for(let i=0;i<DATA.length;i++){const p=DATA[i],x=sx(p[0]),y=sy(p[1]),active=filter==='all'||String(p[4])===filter;screen[i]=[x,y];ctx.globalAlpha=active?0.9:0.1;ctx.fillStyle=color('--point');ctx.beginPath();ctx.arc(x,y,active?2.2:1.2,0,Math.PI*2);ctx.fill();const key=`${Math.floor(x/cell)}:${Math.floor(y/cell)}`;if(!spatial.has(key))spatial.set(key,[]);spatial.get(key).push(i);}
  if(selectedIndex>=0){const p=DATA[selectedIndex];ctx.globalAlpha=1;ctx.fillStyle=color('--bg');ctx.strokeStyle=color('--selected');ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),6,0,Math.PI*2);ctx.fill();ctx.stroke();}
  ctx.globalAlpha=1;
}
function nearest(clientX,clientY){if(!geometry)return null;const rect=canvas.getBoundingClientRect(),x=clientX-rect.left,y=clientY-rect.top,cell=14,cx=Math.floor(x/cell),cy=Math.floor(y/cell);let best=144,index=-1;for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const i of spatial.get(`${cx+dx}:${cy+dy}`)||[]){const px=screen[i][0]-x,py=screen[i][1]-y,dist=px*px+py*py;if(dist<best){best=dist;index=i;}}return index<0?null:{index,x,y};}
canvas.addEventListener('pointermove',event=>{cancelAnimationFrame(hoverFrame);hoverFrame=requestAnimationFrame(()=>{const hit=nearest(event.clientX,event.clientY);if(!hit){tip.style.display='none';return;}const p=DATA[hit.index];tip.textContent=pointText(p);tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=`${Math.max(0,Math.min(plot.clientWidth-w,hit.x+12))}px`;tip.style.top=`${Math.max(0,hit.y-h-10)}px`;});});
canvas.addEventListener('pointerleave',()=>{tip.style.display='none';});
canvas.addEventListener('click',event=>{const hit=nearest(event.clientX,event.clientY);if(!hit)return;selectedIndex=hit.index;selectedInfo.textContent=`已选择：${pointText(DATA[selectedIndex])}`;render();});
layerFilter.addEventListener('change',render);new ResizeObserver(render).observe(plot);render();
</script>
</body>
</html>
"""


def write_interactive_html(
    path: Path, frontier: np.ndarray, metadata: dict[str, object]
) -> None:
    points = [
        [
            round(float(storage), 8),
            round(float(compute), 8),
            int(g1),
            int(g2),
            int(split),
        ]
        for storage, compute, g1, g2, split in frontier
    ]
    html = INTERACTIVE_HTML.replace(
        "__POINTS_JSON__", json.dumps(points, separators=(",", ":"))
    ).replace(
        "__META_JSON__", json.dumps(metadata, separators=(",", ":"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=1_000_000)
    parser.add_argument("--n-max", type=int, default=2_000_000)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8192,
        help="uniform N samples without replacement, including both endpoints",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--dense-gap-max",
        type=int,
        default=128,
        help="enumerate every integer gap up to this value",
    )
    parser.add_argument(
        "--gap-step",
        type=int,
        default=8,
        help="gap spacing above --dense-gap-max",
    )
    parser.add_argument("--storage-limit-gib", type=float, default=1024.0)
    parser.add_argument("--g2-block", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="threads used to evaluate independent g1 rows",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    ns = make_n_samples(args.n_min, args.n_max, args.n_samples, args.seed)
    max_gap = (L - 1) * D + 1
    gaps = make_gap_grid(max_gap, args.dense_gap_max, args.gap_step)
    frontier, stats = evaluate(
        ns, gaps, args.storage_limit_gib, args.g2_block, args.workers
    )
    elapsed = perf_counter() - started

    write_csv(args.output_csv, frontier)
    write_plot(args.output_png, frontier, ns.size)
    metadata = {
        "model": "DeepSeek-V4-Flash",
        "L": L,
        "W": W,
        "N_min": int(ns[0]),
        "N_max": int(ns[-1]),
        "N_samples": int(ns.size),
        "N_sampling": "uniform_without_replacement_with_endpoints",
        "random_seed": args.seed,
        "gap_grid": {
            "dense_integer_range": [1, args.dense_gap_max],
            "coarse_step_above_dense_range": args.gap_step,
            "layer_feasibility_boundaries_always_included": True,
        },
        "storage_limit_gib": args.storage_limit_gib,
        "workers": args.workers,
        "runtime_seconds": elapsed,
        **stats,
    }
    write_interactive_html(args.output_html, frontier, metadata)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"PNG: {args.output_png.resolve()}")
    print(f"HTML: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
