#!/usr/bin/env python3
"""Generate a dependency-free interactive DSV4 checkpoint trade-off chart.

The generated HTML contains every feasible sampled (l, g) point. Hover a point
to inspect l, g, DRAM storage, recovery compute, and Pareto membership. The l/g
sliders select and highlight an exact point without hiding the rest of the data.

Usage:
    python3 docs/dsv4_kv_cache/segmented_checkpoint_gap/plot_checkpoint_pareto.py
    python3 docs/dsv4_kv_cache/segmented_checkpoint_gap/plot_checkpoint_pareto.py --open
    python3 docs/dsv4_kv_cache/segmented_checkpoint_gap/plot_checkpoint_pareto.py --rebuild-data
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from pathlib import Path


L = 43
W = 128
D = W - 1
N = 1_000_000
GIB = 1024**3

# Layer order from DeepSeek-V4-Flash compress_ratios:
# [0, 0, 4, 128, 4, 128, ..., 4].
LAYER_CHECKPOINT_BYTES = [74_880, 74_880]
for _ in range(20):
    LAYER_CHECKPOINT_BYTES.extend([157_824, 600_192])
LAYER_CHECKPOINT_BYTES.append(157_824)
DEFAULT_DATA = Path(__file__).with_name("dsv4_checkpoint_points.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("dsv4_checkpoint_pareto_interactive.html")
DEFAULT_PNG = Path(__file__).with_name("imgs") / "dsv4-checkpoint-pareto-discrete.png"


def recovery_compute(l: int, g: int) -> float:
    """Exact 2g-period average for g1=g and g2=2g under hit constraints."""
    low_removed = 0
    for layer in range(1, l + 1):
        tail = max(g - 1 - layer * D, 0)
        low_removed += tail * (tail + 1) / 2
    low_removed /= g

    high_phase_rows = min(L - l, max((g - 1) // D, 0))
    high_removed = (
        high_phase_rows * g
        - D * high_phase_rows * (high_phase_rows + 1) // 2
    )
    return (
        l * (g - 1) / 2
        - low_removed
        + (L - l) * (2 * g - 1) / 2
        - high_removed / 2
    )


def build_points() -> list[list[float | int]]:
    prefix_bytes = [0]
    for value in LAYER_CHECKPOINT_BYTES:
        prefix_bytes.append(prefix_bytes[-1] + value)

    total_bytes = prefix_bytes[-1]
    raw: list[tuple[float, float, int, int]] = []
    for l in range(1, L):
        # Complete-triangle hit constraints:
        # g - 1 <= l(W - 1), g <= (L - l)(W - 1).
        max_g = min(D * l + 1, D * (L - l))
        low_bytes = prefix_bytes[l]
        high_bytes = total_bytes - low_bytes
        for g in range(1, max_g + 1):
            storage_gib = (
                ((N + g - 1) // g) * low_bytes
                + ((N + 2 * g - 1) // (2 * g)) * high_bytes
            ) / GIB
            raw.append((storage_gib, recovery_compute(l, g), g, l))

    ranked = sorted(raw, key=lambda item: (item[0], item[1]))
    pareto_keys: set[tuple[int, int]] = set()
    best_compute = float("inf")
    for storage, compute, g, l in ranked:
        if compute < best_compute:
            pareto_keys.add((l, g))
            best_compute = compute

    # Compact rows: storage GiB, compute token-layer, g, l, is_pareto.
    return [
        [round(storage, 7), round(compute, 7), g, l, int((l, g) in pareto_keys)]
        for storage, compute, g, l in raw
    ]


def write_points_csv(path: Path, points: list[list[float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["storage_gib", "recovery_token_layers", "g", "l", "is_pareto"]
        )
        writer.writerows(points)


def read_points_csv(path: Path) -> list[list[float | int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "storage_gib",
            "recovery_token_layers",
            "g",
            "l",
            "is_pareto",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(
                f"unexpected CSV columns in {path}; expected {sorted(required)}"
            )
        return [
            [
                float(row["storage_gib"]),
                float(row["recovery_token_layers"]),
                int(row["g"]),
                int(row["l"]),
                int(row["is_pareto"]),
            ]
            for row in reader
        ]


def write_static_plot(path: Path, points: list[list[float | int]]) -> None:
    """Render the g/2g Pareto frontier used by the analysis document."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = sorted((row for row in points if row[4]), key=lambda row: row[0])
    storage = [float(row[0]) for row in frontier]
    compute = [float(row[1]) for row in frontier]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(storage, compute, s=8, color="#2563eb", alpha=0.9, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DRAM checkpoint storage (GiB)")
    ax.set_ylabel("Mean recovery compute (discrete token-layers)")
    ax.set_title("DeepSeek V4 Flash g/2g checkpoint Pareto frontier")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.text(
        0.98,
        0.02,
        f"Pareto points={len(frontier):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#475569",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepSeek V4 Flash checkpoint trade-off</title>
<style>
:root { color-scheme: light dark; --bg:#ffffff; --fg:#172033; --muted:#64748b; --grid:#cbd5e1; --cloud:#64748b; --pareto:#2563eb; --selected:#dc2626; --panel:#f8fafc; --border:#cbd5e1; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --grid:#334155; --cloud:#94a3b8; --pareto:#60a5fa; --selected:#fb7185; --panel:#1e293b; --border:#475569; } }
* { box-sizing:border-box; }
body { margin:0; padding:20px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1280px; margin:0 auto; }
h1 { margin:0 0 14px; font-size:20px; font-weight:600; }
.controls { display:flex; flex-wrap:wrap; gap:18px; margin-bottom:10px; }
label { min-width:220px; flex:1; }
label span { display:block; margin-bottom:4px; }
input[type=range] { width:100%; }
.status { display:flex; flex-wrap:wrap; gap:14px; margin:8px 0; color:var(--muted); }
.plot { position:relative; width:100%; }
canvas { display:block; width:100%; height:650px; }
.tooltip { display:none; position:absolute; pointer-events:none; max-width:260px; padding:8px 10px; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; box-shadow:0 4px 18px rgb(0 0 0 / 18%); }
.legend-dot { font-weight:700; }
.pareto { color:var(--pareto); }
.cloud { color:var(--cloud); }
.selected { color:var(--selected); }
@media (max-width:600px) { body { padding:10px; } canvas { height:480px; } }
</style>
</head>
<body>
<main>
  <h1>DeepSeek V4 Flash checkpoint 存储—计算权衡</h1>
  <div class="controls">
    <label><span>浅层数 l：<output id="l-out">21</output></span><input id="l-input" type="range" min="1" max="42" value="21" step="1"></label>
    <label><span>checkpoint gap g：<output id="g-out">1024</output></span><input id="g-input" type="range" min="1" max="2668" value="1024" step="1"></label>
  </div>
  <div class="status" aria-live="polite">
    <span id="selected-info"></span>
    <span><span class="legend-dot cloud">●</span> 全部可行点</span>
    <span><span class="legend-dot pareto">●</span> Pareto 点</span>
    <span><span class="legend-dot selected">●</span> 当前选择</span>
  </div>
  <div class="plot" id="plot">
    <canvas id="canvas" aria-label="包含所有可行 l 和 g 组合的存储量与计算量散点图"></canvas>
    <div class="tooltip" id="tooltip"></div>
  </div>
</main>
<script>
const DATA = __POINTS_JSON__;
const L = 43, D = 127;
const canvas = document.getElementById('canvas');
const plotNode = document.getElementById('plot');
const tooltip = document.getElementById('tooltip');
const lInput = document.getElementById('l-input');
const gInput = document.getElementById('g-input');
const lOut = document.getElementById('l-out');
const gOut = document.getElementById('g-out');
const selectedInfo = document.getElementById('selected-info');
const ctx = canvas.getContext('2d');
const pointByKey = new Map(DATA.map((p, i) => [`${p[3]}:${p[2]}`, i]));
const frontier = DATA.filter(p => p[4]).sort((a, b) => a[0] - b[0]);
const xLogs = DATA.map(p => Math.log10(p[0]));
const yLogs = DATA.map(p => Math.log10(p[1]));
const xMin = Math.min(...xLogs), xMax = Math.max(...xLogs);
const yMin = Math.min(...yLogs), yMax = Math.max(...yLogs);
let screen = [], spatial = new Map(), geometry = null, hoverFrame = 0;

function color(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function fmt(v, digits=2) { return v.toLocaleString('zh-CN', {maximumFractionDigits:digits}); }
function ticks(minLog, maxLog) {
  const out=[];
  for (let e=Math.floor(minLog); e<=Math.ceil(maxLog); e++) for (const m of [1,2,5]) {
    const value=m*10**e, lv=Math.log10(value);
    if (lv>=minLog && lv<=maxLog) out.push([value,m===1]);
  }
  return out;
}

function selectedPoint() {
  const l=Number(lInput.value);
  const maxG=Math.min(D*l+1,D*(L-l));
  gInput.max=String(maxG);
  if (Number(gInput.value)>maxG) gInput.value=String(maxG);
  return DATA[pointByKey.get(`${l}:${Number(gInput.value)}`)];
}

function render() {
  const width=Math.max(320,plotNode.clientWidth), height=width<600?480:650, ratio=devicePixelRatio||1;
  canvas.style.height=`${height}px`; canvas.width=Math.round(width*ratio); canvas.height=Math.round(height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,width,height);
  const margin={left:width<600?58:78,right:18,top:18,bottom:58};
  const iw=width-margin.left-margin.right, ih=height-margin.top-margin.bottom;
  const sx=v=>margin.left+(Math.log10(v)-xMin)/(xMax-xMin)*iw;
  const sy=v=>margin.top+ih-(Math.log10(v)-yMin)/(yMax-yMin)*ih;
  geometry={sx,sy,width,height,margin,iw,ih};

  ctx.font=`${getComputedStyle(document.body).fontSize} ${getComputedStyle(document.body).fontFamily}`;
  ctx.strokeStyle=color('--grid'); ctx.fillStyle=color('--muted'); ctx.lineWidth=1;
  for (const [v,major] of ticks(xMin,xMax)) { const x=sx(v); ctx.globalAlpha=major?.55:.22; ctx.beginPath();ctx.moveTo(x,margin.top);ctx.lineTo(x,margin.top+ih);ctx.stroke(); if(major){ctx.globalAlpha=1;ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(fmt(v,v<10?1:0),x,margin.top+ih+8);} }
  for (const [v,major] of ticks(yMin,yMax)) { const y=sy(v); ctx.globalAlpha=major?.55:.22; ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(margin.left+iw,y);ctx.stroke(); if(major){ctx.globalAlpha=1;ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(fmt(v,0),margin.left-8,y);} }
  ctx.globalAlpha=1;ctx.fillStyle=color('--fg');ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText('DRAM checkpoint 存储量（GiB，对数）',margin.left+iw/2,height-4);
  ctx.save();ctx.translate(15,margin.top+ih/2);ctx.rotate(-Math.PI/2);ctx.fillText('恢复计算量（token-layer，对数）',0,0);ctx.restore();

  screen=new Array(DATA.length); spatial=new Map(); const cell=14;
  ctx.fillStyle=color('--cloud');ctx.globalAlpha=.18;
  for(let i=0;i<DATA.length;i++) { const p=DATA[i],x=sx(p[0]),y=sy(p[1]);screen[i]=[x,y];ctx.fillRect(x-0.7,y-0.7,1.4,1.4);const key=`${Math.floor(x/cell)}:${Math.floor(y/cell)}`;if(!spatial.has(key))spatial.set(key,[]);spatial.get(key).push(i); }
  ctx.globalAlpha=.9;ctx.fillStyle=color('--pareto');for(const p of frontier){ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),1.8,0,Math.PI*2);ctx.fill();}

  const p=selectedPoint();lOut.value=String(p[3]);gOut.value=String(p[2]);selectedInfo.textContent=`当前：l=${p[3]}，g=${p[2]}，${fmt(p[0],3)} GiB，${fmt(p[1],0)} token-layer${p[4]?'，Pareto 点':'，非 Pareto 点'}`;
  ctx.globalAlpha=1;ctx.fillStyle=color('--bg');ctx.strokeStyle=color('--selected');ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),6,0,Math.PI*2);ctx.fill();ctx.stroke();
}

function nearest(clientX,clientY) {
  if(!geometry)return null;const rect=canvas.getBoundingClientRect(),x=clientX-rect.left,y=clientY-rect.top,cell=14,cx=Math.floor(x/cell),cy=Math.floor(y/cell);let best=100,idx=-1;
  for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){const bucket=spatial.get(`${cx+dx}:${cy+dy}`)||[];for(const i of bucket){const px=screen[i][0]-x,py=screen[i][1]-y,dist=px*px+py*py;if(dist<best){best=dist;idx=i;}}}
  return idx<0?null:{point:DATA[idx],x,y};
}

canvas.addEventListener('pointermove',e=>{cancelAnimationFrame(hoverFrame);hoverFrame=requestAnimationFrame(()=>{const hit=nearest(e.clientX,e.clientY);if(!hit){tooltip.style.display='none';return;}const p=hit.point;tooltip.innerHTML=`<b>l=${p[3]}，g=${p[2]}</b><br>${fmt(p[0],4)} GiB<br>${fmt(p[1],2)} token-layer<br>${p[4]?'Pareto 最优':'非 Pareto 最优'}`;tooltip.style.display='block';const w=tooltip.offsetWidth,h=tooltip.offsetHeight;tooltip.style.left=`${Math.max(0,Math.min(plotNode.clientWidth-w,hit.x+12))}px`;tooltip.style.top=`${Math.max(0,hit.y-h-10)}px`;});});
canvas.addEventListener('pointerleave',()=>{tooltip.style.display='none';});
canvas.addEventListener('click',e=>{const hit=nearest(e.clientX,e.clientY);if(!hit)return;lInput.value=String(hit.point[3]);gInput.value=String(hit.point[2]);render();});
lInput.addEventListener('input',render);gInput.addEventListener('input',render);new ResizeObserver(render).observe(plotNode);render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="input CSV table (default: %(default)s)",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_PNG,
        help="output static PNG path (default: %(default)s)",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="recompute all points and overwrite --data before rendering",
    )
    parser.add_argument("--open", action="store_true", help="open the generated chart")
    args = parser.parse_args()

    if args.rebuild_data:
        write_points_csv(args.data, build_points())
    if not args.data.exists():
        parser.error(f"data table not found: {args.data}; run with --rebuild-data")

    points = read_points_csv(args.data)
    payload = json.dumps(points, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__POINTS_JSON__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    write_static_plot(args.output_png, points)
    print(
        f"wrote {args.output.resolve()}\n"
        f"wrote {args.output_png.resolve()}\n"
        f"points={len(points)}, pareto={sum(row[4] for row in points)}"
    )
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
