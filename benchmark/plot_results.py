#!/usr/bin/env python3
"""Generate poster-ready charts from benchmark JSON results.

Reads all ``*__*.json`` files in ``results/`` and produces:
  1. Grouped bar chart  - Mean wall-clock time per query
  2. Grouped bar chart  - Peak VRAM usage
  3. Heatmap / table    - Task accuracy scores
  4. Speedup bar chart  - Relative speedup from optimizations

Usage:
    python plot_results.py [--results-dir results] [--output-dir results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PyTorch-themed color palette
# ---------------------------------------------------------------------------

TORCH_ORANGE = "#EE4C2C"
TORCH_DARK = "#1A2744"
TORCH_BLUE = "#1B3A5C"
TORCH_LIGHT_BLUE = "#4A90D9"
TORCH_GREEN = "#1A7A3A"
TORCH_GRAY = "#6C7A89"

CONFIG_COLORS = {
    "baseline": TORCH_GRAY,
    "prefix-cache": TORCH_LIGHT_BLUE,
    "prefix-cache-batched": TORCH_ORANGE,
}

CONFIG_LABELS = {
    "baseline": "Baseline",
    "prefix-cache": "+ Prefix Cache",
    "prefix-cache-batched": "+ Prefix Cache\n+ Batched Sub-calls",
}

MODEL_SHORT = {
    "Qwen/Qwen3-8B": "Qwen3-8B\n(base)",
    "mit-oasys/rlm-qwen3-8b-v0.1": "RLM-Qwen3-8B\n(fine-tuned)",
}


def _short_model(name: str) -> str:
    return MODEL_SHORT.get(name, name.split("/")[-1])


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Return mapping of (model, config) -> aggregated result dict."""
    data = {}
    for path in sorted(results_dir.glob("*__*.json")):
        with open(path) as f:
            obj = json.load(f)
        key = (obj["model"], obj["config"])
        data[key] = obj
    return data


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes, title: str, ylabel: str):
    ax.set_title(title, fontsize=14, fontweight="bold", color=TORCH_DARK, pad=12)
    ax.set_ylabel(ylabel, fontsize=11, color=TORCH_DARK)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def _add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.1f}"):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=TORCH_DARK,
            )


# ---------------------------------------------------------------------------
# Chart 1: Wall-clock time
# ---------------------------------------------------------------------------

def plot_wall_clock(data: dict, output_dir: Path):
    models = sorted({m for m, _ in data})
    configs = ["baseline", "prefix-cache", "prefix-cache-batched"]
    present_configs = [c for c in configs if any((m, c) in data for m in models)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.22
    offset = -(len(present_configs) - 1) / 2 * width

    for i, cfg in enumerate(present_configs):
        vals = []
        for m in models:
            agg = data.get((m, cfg), {}).get("aggregates", {})
            vals.append(agg.get("mean_wall_clock_s", 0))
        bars = ax.bar(
            x + offset + i * width,
            vals,
            width * 0.88,
            label=CONFIG_LABELS.get(cfg, cfg),
            color=CONFIG_COLORS.get(cfg, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )
        _add_bar_labels(ax, bars, "{:.1f}s")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models])
    _apply_style(ax, "Mean Wall-Clock Time per Query", "Seconds")
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    out = output_dir / "chart_wall_clock.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Chart 2: Peak VRAM
# ---------------------------------------------------------------------------

def plot_peak_vram(data: dict, output_dir: Path):
    models = sorted({m for m, _ in data})
    configs = ["baseline", "prefix-cache", "prefix-cache-batched"]
    present_configs = [c for c in configs if any((m, c) in data for m in models)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.22
    offset = -(len(present_configs) - 1) / 2 * width

    for i, cfg in enumerate(present_configs):
        vals = []
        for m in models:
            agg = data.get((m, cfg), {}).get("aggregates", {})
            vals.append(agg.get("max_peak_vram_mib", 0) / 1024)  # GiB
        bars = ax.bar(
            x + offset + i * width,
            vals,
            width * 0.88,
            label=CONFIG_LABELS.get(cfg, cfg),
            color=CONFIG_COLORS.get(cfg, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )
        _add_bar_labels(ax, bars, "{:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models])
    _apply_style(ax, "Peak GPU VRAM Usage", "GiB")
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    out = output_dir / "chart_peak_vram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Chart 3: Task accuracy heatmap
# ---------------------------------------------------------------------------

def plot_accuracy_table(data: dict, output_dir: Path):
    models = sorted({m for m, _ in data})
    configs = ["baseline", "prefix-cache", "prefix-cache-batched"]
    present_configs = [c for c in configs if any((m, c) in data for m in models)]

    matrix = []
    for m in models:
        row = []
        for cfg in present_configs:
            agg = data.get((m, cfg), {}).get("aggregates", {})
            row.append(agg.get("mean_score", 0) * 100)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    col_labels = [CONFIG_LABELS.get(c, c).replace("\n", " ") for c in present_configs]
    row_labels = [_short_model(m).replace("\n", " ") for m in models]

    cell_text = [[f"{v:.1f}%" for v in row] for row in matrix]

    cell_colors = []
    vmin, vmax = matrix.min(), max(matrix.max(), 1)
    for row in matrix:
        row_colors = []
        for v in row:
            intensity = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            r = 1.0 - intensity * 0.4
            g = 1.0 - intensity * 0.15
            b = 1.0 - intensity * 0.4
            row_colors.append((r, g, b))
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_text_props(fontweight="bold", color=TORCH_DARK)
            cell.set_facecolor("#F0F4FA")
        if col == -1:
            cell.set_text_props(fontweight="bold", color=TORCH_DARK)

    ax.set_title(
        "Task Accuracy (OOLONG-synth)",
        fontsize=14,
        fontweight="bold",
        color=TORCH_DARK,
        pad=20,
    )

    fig.tight_layout()
    out = output_dir / "chart_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Chart 4: Speedup from optimizations
# ---------------------------------------------------------------------------

def plot_speedup(data: dict, output_dir: Path):
    models = sorted({m for m, _ in data})
    configs_to_compare = ["prefix-cache", "prefix-cache-batched"]
    present = [c for c in configs_to_compare if any((m, c) in data for m in models)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.30
    offset = -(len(present) - 1) / 2 * width

    for i, cfg in enumerate(present):
        speedups = []
        for m in models:
            base_time = data.get((m, "baseline"), {}).get("aggregates", {}).get("mean_wall_clock_s", 1)
            opt_time = data.get((m, cfg), {}).get("aggregates", {}).get("mean_wall_clock_s", 1)
            speedups.append(base_time / opt_time if opt_time > 0 else 1.0)
        bars = ax.bar(
            x + offset + i * width,
            speedups,
            width * 0.88,
            label=CONFIG_LABELS.get(cfg, cfg),
            color=CONFIG_COLORS.get(cfg, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )
        _add_bar_labels(ax, bars, "{:.2f}x")

    ax.axhline(y=1.0, color=TORCH_GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in models])
    _apply_style(ax, "Speedup vs Baseline", "Speedup Factor")
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    out = output_dir / "chart_speedup.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Chart 5 (bonus): Iterations & sub-calls comparison
# ---------------------------------------------------------------------------

def plot_iterations_subcalls(data: dict, output_dir: Path):
    models = sorted({m for m, _ in data})
    configs = ["baseline", "prefix-cache", "prefix-cache-batched"]
    present_configs = [c for c in configs if any((m, c) in data for m in models)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    width = 0.22
    offset = -(len(present_configs) - 1) / 2 * width

    for i, cfg in enumerate(present_configs):
        iters_vals = []
        sc_vals = []
        for m in models:
            agg = data.get((m, cfg), {}).get("aggregates", {})
            iters_vals.append(agg.get("mean_iterations", 0))
            sc_vals.append(agg.get("mean_subcalls", 0))

        color = CONFIG_COLORS.get(cfg, "#999")
        label = CONFIG_LABELS.get(cfg, cfg)

        bars1 = ax1.bar(x + offset + i * width, iters_vals, width * 0.88, label=label, color=color, edgecolor="white", linewidth=0.5)
        _add_bar_labels(ax1, bars1, "{:.1f}")

        bars2 = ax2.bar(x + offset + i * width, sc_vals, width * 0.88, label=label, color=color, edgecolor="white", linewidth=0.5)
        _add_bar_labels(ax2, bars2, "{:.1f}")

    for ax, title, ylabel in [
        (ax1, "Mean REPL Iterations per Query", "Iterations"),
        (ax2, "Mean Sub-calls per Query", "Sub-calls"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([_short_model(m) for m in models])
        _apply_style(ax, title, ylabel)

    ax1.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = output_dir / "chart_iterations_subcalls.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate poster charts from benchmark results")
    parser.add_argument("--results-dir", default="results", help="Directory containing JSON result files")
    parser.add_argument("--output-dir", default="results", help="Directory for output PNGs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    print("Loading results ...")
    data = load_results(results_dir)

    if not data:
        print(f"ERROR: No result files found in {results_dir}/")
        print("Run benchmarks first:  python run_benchmark.py --model ... --config ...")
        return

    print(f"Found {len(data)} result file(s):")
    for (model, cfg) in sorted(data):
        n = data[(model, cfg)].get("num_samples", "?")
        print(f"  {model} / {cfg}  ({n} samples)")

    print("\nGenerating charts ...")
    plot_wall_clock(data, output_dir)
    plot_peak_vram(data, output_dir)
    plot_accuracy_table(data, output_dir)
    plot_speedup(data, output_dir)
    plot_iterations_subcalls(data, output_dir)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
