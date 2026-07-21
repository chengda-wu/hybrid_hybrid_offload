#!/usr/bin/env python3
"""Probe whether three ordered checkpoint gaps improve on two gaps.

Layers ``1..l1`` use ``g1``, layers ``l1+1..l2`` use ``g2``, and the
remaining layers use ``g3``, where ``g1 <= g2 <= g3``.  For every sampled N:

    r1 = N mod g1;       c1 = N - r1
    r2 = c1 mod g2;      c2 = c1 - r2
    r3 = c2 mod g3

    C = l1*r1 - r1^2/(2D)
        + (l2-l1)*g2 - r2^2/(2D)
        + (L-l2)*g3 - r3^2/(2D)

This is deliberately a probe rather than an exhaustive five-dimensional
search.  It samples split/ratio shapes, chooses the smallest common gap scale
meeting each storage budget, ranks them with a cheap phase approximation, and
then evaluates the most promising candidates with the same exact N samples as
the two-gap experiment.  Repeated-gap boundary candidates are injected for a
fair comparison, but they reproduce the two-gap baseline only when both new
segments independently satisfy the checkpoint-hit constraint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import plot_checkpoint_g1_g2_pareto as two


HERE = Path(__file__).resolve().parent
DEFAULT_TWO_GAP_CSV = HERE / "dsv4_checkpoint_g1_g2_pareto.csv"
DEFAULT_OUTPUT = HERE / "dsv4_checkpoint_g1_g2_g3_probe.csv"
DEFAULT_BUDGETS = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)


def load_two_gap_frontier(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"unexpected two-gap frontier format: {path}")
    return data


def gap_statistics(ns: np.ndarray, max_gap: int) -> tuple[np.ndarray, ...]:
    """Precompute exact sampled statistics indexed directly by gap."""
    mean_ceil = np.zeros(max_gap + 1, dtype=np.float64)
    mean_r = np.zeros(max_gap + 1, dtype=np.float64)
    mean_r_sq = np.zeros(max_gap + 1, dtype=np.float64)
    for gap in range(1, max_gap + 1):
        phase = ns % gap
        mean_ceil[gap] = np.mean((ns + gap - 1) // gap)
        mean_r[gap] = np.mean(phase)
        mean_r_sq[gap] = np.mean(phase.astype(np.float64) ** 2)
    return mean_ceil, mean_r, mean_r_sq


def storage_gib(
    gaps: tuple[int, int, int],
    splits: tuple[int, int],
    mean_ceil: np.ndarray,
    prefix_bytes: np.ndarray,
) -> float:
    g1, g2, g3 = gaps
    l1, l2 = splits
    values = (
        prefix_bytes[l1] * mean_ceil[g1]
        + (prefix_bytes[l2] - prefix_bytes[l1]) * mean_ceil[g2]
        + (prefix_bytes[-1] - prefix_bytes[l2]) * mean_ceil[g3]
    )
    return float(values / two.GIB)


def scaled_candidate(
    ratios: tuple[float, float, float],
    splits: tuple[int, int],
    budget: float,
    mean_ceil: np.ndarray,
    prefix_bytes: np.ndarray,
) -> tuple[float, int, int, int, int, int] | None:
    """Find the smallest rounded common scale satisfying a storage budget."""
    l1, l2 = splits
    widths = (l1, l2 - l1, two.L - l2)
    maxima = np.asarray([width * two.D + 1 for width in widths])
    ratio_array = np.asarray(ratios, dtype=np.float64)
    high = int(np.min(maxima / ratio_array))
    if high < 1:
        return None

    def gaps_at(scale: int) -> tuple[int, int, int]:
        values = np.maximum(1, np.rint(scale * ratio_array).astype(np.int64))
        values = np.maximum.accumulate(values)
        return int(values[0]), int(values[1]), int(values[2])

    high_gaps = gaps_at(high)
    if any(gap > maximum for gap, maximum in zip(high_gaps, maxima)):
        high -= 1
        if high < 1:
            return None
        high_gaps = gaps_at(high)
    if storage_gib(high_gaps, splits, mean_ceil, prefix_bytes) > budget:
        return None

    low = 1
    while low < high:
        mid = (low + high) // 2
        gaps = gaps_at(mid)
        if storage_gib(gaps, splits, mean_ceil, prefix_bytes) <= budget:
            high = mid
        else:
            low = mid + 1
    gaps = gaps_at(low)
    if any(gap > maximum for gap, maximum in zip(gaps, maxima)):
        return None
    storage = storage_gib(gaps, splits, mean_ceil, prefix_bytes)
    return storage, *gaps, l1, l2


def approximate_compute(
    candidates: np.ndarray,
    mean_r: np.ndarray,
    mean_r_sq: np.ndarray,
) -> np.ndarray:
    """Rank candidates cheaply; later selection uses exact chained phases."""
    g1 = candidates[:, 1].astype(np.int64)
    g2 = candidates[:, 2]
    g3 = candidates[:, 3]
    l1 = candidates[:, 4]
    l2 = candidates[:, 5]
    return (
        l1 * mean_r[g1]
        - mean_r_sq[g1] / (2 * two.D)
        + (l2 - l1) * g2
        - (g2 - 1) * (2 * g2 - 1) / (12 * two.D)
        + (two.L - l2) * g3
        - (g3 - 1) * (2 * g3 - 1) / (12 * two.D)
    )


def exact_compute(
    ns: np.ndarray, candidates: np.ndarray, block_size: int
) -> np.ndarray:
    result = np.empty(candidates.shape[0], dtype=np.float64)
    ns_column = ns[:, None]
    for start in range(0, candidates.shape[0], block_size):
        stop = min(candidates.shape[0], start + block_size)
        block = candidates[start:stop]
        g1 = block[:, 1].astype(np.int64)[None, :]
        g2 = block[:, 2].astype(np.int64)[None, :]
        g3 = block[:, 3].astype(np.int64)[None, :]
        l1 = block[:, 4][None, :]
        l2 = block[:, 5][None, :]
        r1 = ns_column % g1
        c1 = ns_column - r1
        r2 = c1 % g2
        c2 = c1 - r2
        r3 = c2 % g3
        compute = (
            l1 * r1
            - r1.astype(np.float64) ** 2 / (2 * two.D)
            + (l2 - l1) * g2
            - r2.astype(np.float64) ** 2 / (2 * two.D)
            + (two.L - l2) * g3
            - r3.astype(np.float64) ** 2 / (2 * two.D)
        )
        result[start:stop] = np.mean(compute, axis=0)
    return result


def generate_candidates(
    rng: np.random.Generator,
    budget: float,
    baseline: np.ndarray,
    count: int,
    mean_ceil: np.ndarray,
    prefix_bytes: np.ndarray,
) -> np.ndarray:
    base_g1, base_g3, base_split = map(int, baseline[2:5])
    candidates: set[tuple[float, int, int, int, int, int]] = set()

    def add_if_feasible(
        gaps: tuple[int, int, int], splits: tuple[int, int]
    ) -> None:
        l1, l2 = splits
        widths = (l1, l2 - l1, two.L - l2)
        if all(gap - 1 <= width * two.D for gap, width in zip(gaps, widths)):
            storage = storage_gib(gaps, splits, mean_ceil, prefix_bytes)
            candidates.add((storage, *gaps, *splits))

    # Repeated-gap candidates at both sides of the old split.  They are exact
    # embeddings only when the extra segment is itself wide enough to hit.
    for l1 in range(1, base_split):
        gaps = (base_g1, base_g1, base_g3)
        splits = (l1, base_split)
        add_if_feasible(gaps, splits)
    for l2 in range(base_split + 1, two.L):
        gaps = (base_g1, base_g3, base_g3)
        splits = (base_split, l2)
        add_if_feasible(gaps, splits)

    # The low-storage corner has g3/g1 above 30 and is easy to miss with
    # ratio sampling.  Exhaust the two one-layer leading segments there.
    if budget <= 16.0:
        splits = (1, 2)
        max_g3 = (two.L - 2) * two.D + 1
        for g1 in range(1, two.D + 2):
            for g2 in range(g1, two.D + 2):
                low, high = g2, max_g3
                if storage_gib((g1, g2, high), splits, mean_ceil, prefix_bytes) > budget:
                    continue
                while low < high:
                    mid = (low + high) // 2
                    if storage_gib((g1, g2, mid), splits, mean_ceil, prefix_bytes) <= budget:
                        high = mid
                    else:
                        low = mid + 1
                candidates.add(
                    (
                        storage_gib((g1, g2, low), splits, mean_ceil, prefix_bytes),
                        g1,
                        g2,
                        low,
                        *splits,
                    )
                )

    base_ratio = max(1.0, base_g3 / base_g1)
    for index in range(count):
        mode = index % 4
        if mode == 0 and base_split > 1:
            l2 = base_split
            l1 = int(rng.integers(1, l2))
        elif mode == 1 and base_split < two.L - 1:
            l1 = base_split
            l2 = int(rng.integers(l1 + 1, two.L))
        else:
            l1, l2 = sorted(rng.choice(np.arange(1, two.L), 2, replace=False))
            l1, l2 = int(l1), int(l2)

        if mode < 2:
            q3 = base_ratio * float(np.exp(rng.normal(0.0, 0.12)))
            q3 = max(1.0, min(4.0, q3))
            q2 = float(np.exp(rng.uniform(0.0, np.log(q3))))
        elif mode == 2:
            q2 = float(np.exp(rng.uniform(0.0, np.log(1.5))))
            q3 = q2 * float(np.exp(rng.uniform(0.0, np.log(1.5))))
        else:
            # Broad tail for very small storage budgets, where a one-layer
            # first/second segment can force g1,g2 near 128 while g3 is >5000.
            q2 = float(np.exp(rng.uniform(0.0, np.log(4.0))))
            q3 = q2 * float(np.exp(rng.uniform(0.0, np.log(64.0))))

        candidate = scaled_candidate(
            (1.0, q2, q3),
            (l1, l2),
            budget,
            mean_ceil,
            prefix_bytes,
        )
        if candidate is not None:
            candidates.add(candidate)
    if not candidates:
        return np.empty((0, 6), dtype=np.float64)
    return np.asarray(sorted(candidates), dtype=np.float64)


def probe_budget(
    rng: np.random.Generator,
    ns: np.ndarray,
    budget: float,
    baseline: np.ndarray,
    mean_ceil: np.ndarray,
    mean_r: np.ndarray,
    mean_r_sq: np.ndarray,
    prefix_bytes: np.ndarray,
    random_candidates: int,
    exact_candidates: int,
    block_size: int,
) -> tuple[np.ndarray | None, int, int]:
    candidates = generate_candidates(
        rng,
        budget,
        baseline,
        random_candidates,
        mean_ceil,
        prefix_bytes,
    )
    if candidates.shape[0] == 0:
        return None, 0, 0
    approx = approximate_compute(candidates, mean_r, mean_r_sq)
    selected_count = min(exact_candidates, candidates.shape[0])
    selected = np.argpartition(approx, selected_count - 1)[:selected_count]
    base_g1, base_g2, base_split = map(int, baseline[2:5])
    embedding_mask = (
        (
            (candidates[:, 1] == base_g1)
            & (candidates[:, 2] == base_g1)
            & (candidates[:, 3] == base_g2)
            & (candidates[:, 5] == base_split)
        )
        | (
            (candidates[:, 1] == base_g1)
            & (candidates[:, 2] == base_g2)
            & (candidates[:, 3] == base_g2)
            & (candidates[:, 4] == base_split)
        )
    )
    selected = np.unique(
        np.concatenate((selected, np.flatnonzero(embedding_mask)))
    )
    finalists = candidates[selected]
    exact = exact_compute(ns, finalists, block_size)
    best = finalists[int(np.argmin(exact))]
    return (
        np.concatenate((best, [float(np.min(exact))])),
        candidates.shape[0],
        selected.size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-gap-csv", type=Path, default=DEFAULT_TWO_GAP_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="candidate-search seed; defaults to --seed + 3",
    )
    parser.add_argument("--random-candidates", type=int, default=320_000)
    parser.add_argument("--exact-candidates", type=int, default=15_000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = two.make_n_samples(1_000_000, 2_000_000, args.n_samples, args.seed)
    max_gap = (two.L - 1) * two.D + 1
    mean_ceil, mean_r, mean_r_sq = gap_statistics(ns, max_gap)
    prefix_bytes = np.concatenate(
        ([0], np.cumsum(two.LAYER_CHECKPOINT_BYTES, dtype=np.int64))
    )
    two_gap = load_two_gap_frontier(args.two_gap_csv)
    search_seed = args.seed + 3 if args.search_seed is None else args.search_seed
    rng = np.random.default_rng(search_seed)

    rows = []
    for budget in args.budgets:
        feasible = two_gap[two_gap[:, 0] <= budget]
        if feasible.size == 0:
            raise ValueError(f"no two-gap baseline at {budget} GiB")
        baseline = feasible[int(np.argmin(feasible[:, 1]))]
        best, generated, exact_count = probe_budget(
            rng,
            ns,
            budget,
            baseline,
            mean_ceil,
            mean_r,
            mean_r_sq,
            prefix_bytes,
            args.random_candidates,
            args.exact_candidates,
            args.block_size,
        )
        if best is None:
            row = [budget, *baseline, *("" for _ in range(8)), generated, exact_count]
            print(
                f"{budget:7.1f} GiB: two={baseline[1]:10.3f}, "
                "three=not found (no feasible sampled shape)"
            )
            rows.append(row)
            continue
        improvement = (baseline[1] - best[6]) / baseline[1] * 100
        row = [budget, *baseline, *best, improvement, generated, exact_count]
        rows.append(row)
        print(
            f"{budget:7.1f} GiB: two={baseline[1]:10.3f}, "
            f"three={best[6]:10.3f}, improvement={improvement:6.3f}% "
            f"at g=({int(best[1])},{int(best[2])},{int(best[3])}), "
            f"l=({int(best[4])},{int(best[5])})"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "budget_gib",
                "two_storage_gib",
                "two_compute",
                "two_g1",
                "two_g2",
                "two_l",
                "three_storage_gib",
                "three_g1",
                "three_g2",
                "three_g3",
                "three_l1",
                "three_l2",
                "three_compute",
                "improvement_percent",
                "generated_candidates",
                "exact_candidates",
            ]
        )
        writer.writerows(rows)
    print(f"CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
