#!/usr/bin/env python3
"""Render KAITO-style architecture diagrams (SVG + PNG)."""

from __future__ import annotations

from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parent

# Sampled from kaito-project/kaito website/static/img/arch.png
BG = "#F2F2F2"
GROUP = "#FBFBFB"
GROUP_STROKE = "#C8C8C8"
PINK = "#E098D8"
PINK_INK = "#6A1858"
PINK_STROKE = "#8A2A7A"
GREEN = "#B0E0A0"
GREEN_INK = "#1E4A14"
GREEN_STROKE = "#3A7A28"
BLUE = "#0898D0"
BLUE_STROKE = "#045878"
ORANGE = "#F0C0A8"
ORANGE_INK = "#6A2E10"
ORANGE_STROKE = "#C06028"
PURPLE = "#8890E0"
PURPLE_STROKE = "#3A4088"
GRAY_STROKE = "#6E6E6E"
INK = "#1A1A1A"
MUTED = "#555555"
WHITE = "#FFFFFF"
FONT = "Inter, Liberation Sans, DejaVu Sans, sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill, stroke, sw=2.4, r=18, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def txt(x, y, s, size=18, weight=700, fill=INK, anchor="middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}">{esc(s)}</text>'
    )


def marker(mid: str, color: str) -> str:
    return (
        f'<marker id="{mid}" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        f'orient="auto" markerUnits="strokeWidth">'
        f'<polygon points="0 0, 10 4, 0 8" fill="{color}"/></marker>'
    )


def defs() -> str:
    return (
        "<defs>"
        + marker("arr", INK)
        + marker("arr-p", PINK_STROKE)
        + marker("arr-g", GREEN_STROKE)
        + marker("arr-b", BLUE_STROKE)
        + marker("arr-o", ORANGE_STROKE)
        + marker("arr-u", PURPLE_STROKE)
        + "</defs>"
    )


def line(x1, y1, x2, y2, color=INK, sw=2.2, m="arr", dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{m})"' if m else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw}"{d}{mk}/>'
    )


def path(d, color=INK, sw=2.2, m="arr", dash=None) -> str:
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{m})"' if m else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{mk}/>'


def badge(cx, cy, fill, stroke, glyph: str) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<text x="{cx}" y="{cy + 4.2}" text-anchor="middle" fill="{stroke}" '
        f'font-family="{FONT}" font-size="12" font-weight="800">{esc(glyph)}</text>'
    )


def pill(x, y, w, h, fill, stroke, label, ink=WHITE, sw=1.4) -> str:
    return rect(x, y, w, h, fill, stroke, sw=sw, r=h / 2) + txt(
        x + w / 2, y + h * 0.68, label, size=13, weight=800, fill=ink
    )


def render(svg: str, name: str, w: int, h: int) -> None:
    svg_path = OUT / f"{name}.svg"
    png_path = OUT / f"{name}.png"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=w * 2,
        output_height=h * 2,
        background_color=BG,
    )
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes, {w * 2}x{h * 2})")


def system_arch() -> None:
    """Overview diagram — same visual grammar as KAITO arch.png."""
    W, H = 1760, 1040
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs(),
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
    ]

    # External dataset (dashed, like External Gateway)
    parts += [
        rect(40, 40, 260, 108, WHITE, GRAY_STROKE, sw=2.2, r=16, dash="7 5"),
        badge(68, 68, WHITE, GRAY_STROKE, "D"),
        txt(178, 78, "OOLONG-synth", 19, 800),
        txt(178, 104, "Long-context tasks", 13.5, 600, MUTED),
        txt(178, 126, "Hugging Face dataset", 13.5, 600, MUTED),
    ]

    # Conference layer (pink, like InferenceSet)
    parts += [
        rect(340, 28, 1080, 132, PINK, PINK_STROKE, sw=2.6, r=20),
        badge(372, 58, WHITE, PINK_STROKE, "+"),
        txt(880, 64, "This repository", 24, 800, PINK_INK),
        txt(880, 92, "Conference layer  ·  PyTorch Conference EU 2026", 14.5, 600, PINK_INK),
        pill(390, 108, 300, 36, WHITE, PINK_STROKE, "run_benchmark.py", PINK_INK),
        pill(720, 108, 300, 36, WHITE, PINK_STROKE, "plot_results.py  ·  poster", PINK_INK),
        pill(1050, 108, 330, 36, WHITE, PINK_STROKE, "serve_model.sh  ·  setup", PINK_INK),
    ]

    # Metrics (orange, like AutoScaler)
    parts += [
        rect(1460, 40, 260, 108, ORANGE, ORANGE_STROKE, sw=2.2, r=16),
        badge(1488, 68, WHITE, ORANGE_STROKE, "M"),
        txt(1600, 78, "Metrics", 19, 800, ORANGE_INK),
        txt(1600, 104, "Wall-clock  ·  VRAM", 13.5, 600, ORANGE_INK),
        txt(1600, 126, "Tokens  ·  iters  ·  sub-calls", 13.5, 600, ORANGE_INK),
    ]

    # Down into runtime
    parts += [
        path("M 170 148 V 188 H 360", GREEN_STROKE, 2.2, "arr-g"),
        line(880, 160, 880, 188, PINK_STROKE, 2.4, "arr-p"),
        path("M 1590 148 V 188 H 1400", ORANGE_STROKE, 2.2, "arr-o", "6 4"),
        txt(210, 176, "samples", 12, 700, GREEN_STROKE, "start"),
        txt(1540, 176, "observe", 12, 700, ORANGE_STROKE, "end"),
    ]

    # Runtime group (like InferencePool)
    parts += [
        rect(40, 196, 1680, 500, GROUP, GROUP_STROKE, 2.0, 22),
        txt(64, 226, "RLM Runtime", 14, 800, MUTED, "start"),
        txt(186, 226, "upstream  ·  alexzhang13/rlm  (rlms)", 13, 600, "#808080", "start"),
    ]

    # Orchestrator
    parts += [
        rect(400, 248, 960, 88, PINK, PINK_STROKE, 2.4, 18),
        badge(432, 276, WHITE, PINK_STROKE, "R"),
        txt(880, 280, "RLM Orchestrator", 22, 800, PINK_INK),
        txt(880, 308, "hist  →  generate  →  exec  →  trim  →  repeat until Final", 14, 600, PINK_INK),
    ]
    parts.append(line(880, 336, 880, 364, PINK_STROKE, 2.2, "arr-p"))

    # Two green workspaces
    parts += [
        rect(72, 376, 720, 132, GREEN, GREEN_STROKE, 2.4, 18),
        badge(104, 404, WHITE, GREEN_STROKE, "P"),
        txt(432, 412, "Python REPL", 22, 800, GREEN_INK),
        txt(432, 442, "Prompt P as a variable   ·   sub_RLM() hook", 14, 600, GREEN_INK),
        txt(432, 466, "Peek  ·  grep  ·  chunk  ·  recurse  ·  set Final", 14, 600, GREEN_INK),
        rect(968, 376, 720, 132, GREEN, GREEN_STROKE, 2.4, 18),
        badge(1000, 404, WHITE, GREEN_STROKE, "L"),
        txt(1328, 412, "Language Model", 22, 800, GREEN_INK),
        txt(1328, 442, "Qwen3-8B   or   RLM-Qwen3-8B", 14, 600, GREEN_INK),
        txt(1328, 466, "Sees hist only  ·  never the raw prompt P", 14, 600, GREEN_INK),
    ]

    # Loop arrows
    parts += [
        path("M 792 416 H 956", BLUE_STROKE, 2.4, "arr-b"),
        path("M 956 468 H 792", GREEN_STROKE, 2.4, "arr-g"),
        txt(874, 404, "metadata", 12, 800, BLUE_STROKE),
        txt(874, 490, "Python code", 12, 800, GREEN_STROKE),
    ]

    parts += [
        line(432, 508, 432, 536, GREEN_STROKE, 2.2, "arr-g"),
        line(1328, 508, 1328, 536, GREEN_STROKE, 2.2, "arr-g"),
    ]

    # Blue workloads
    parts += [
        rect(72, 548, 720, 116, BLUE, BLUE_STROKE, 2.4, 18),
        badge(104, 576, WHITE, BLUE_STROKE, "F"),
        txt(432, 586, "REPL result", 20, 800, WHITE),
        txt(432, 616, "Final set?   Yes → output    No → loop", 14, 600, "#E8F6FC"),
        rect(968, 548, 720, 116, BLUE, BLUE_STROKE, 2.4, 18),
        badge(1000, 576, WHITE, BLUE_STROKE, "V"),
        txt(1328, 586, "vLLM inference", 20, 800, WHITE),
        txt(1328, 616, "OpenAI HTTP   ·   localhost:8000/v1   ·   bf16", 14, 600, "#E8F6FC"),
    ]
    parts += [
        path("M 792 606 H 956", ORANGE_STROKE, 2.0, "arr-o", "5 4"),
        txt(874, 594, "sub_RLM()", 12, 800, ORANGE_STROKE),
    ]

    # Configs (purple, like NodePool)
    parts += [
        path("M 432 664 V 700 H 1328", PURPLE_STROKE, 2.2, None),
        line(432, 700, 432, 724, PURPLE_STROKE, 2.2, "arr-u"),
        line(880, 700, 880, 724, PURPLE_STROKE, 2.2, "arr-u"),
        line(1328, 700, 1328, 724, PURPLE_STROKE, 2.2, "arr-u"),
    ]

    configs = [
        (72, "baseline", "Prefix cache off", "1 sequential sub-call"),
        (616, "prefix-cache", "Prefix cache on", "1 sequential sub-call"),
        (1160, "prefix-cache-batched", "Prefix cache on", "4 parallel sub-calls"),
    ]
    for x, title, a, b in configs:
        parts += [
            rect(x, 736, 528, 100, PURPLE, PURPLE_STROKE, 2.4, 16),
            txt(x + 264, 774, title, 20, 800, WHITE),
            txt(x + 264, 804, f"{a}   ·   {b}", 13.5, 600, "#EEF0FF"),
        ]

    # Hardware bar
    parts += [
        line(432, 836, 432, 868, INK, 2.2, "arr"),
        line(880, 836, 880, 868, INK, 2.2, "arr"),
        line(1328, 836, 1328, 868, INK, 2.2, "arr"),
        rect(40, 880, 1680, 124, WHITE, INK, 2.8, 18),
        txt(880, 932, "NVIDIA H100  80GB", 28, 800),
        txt(880, 972, "CUDA   ·   PyTorch   ·   vLLM   ·   bfloat16", 16, 600, MUTED),
    ]

    # Metrics dashed down the right, inside the canvas
    parts += [
        path("M 1704 148 V 606 H 1688", ORANGE_STROKE, 2.0, "arr-o", "6 4"),
        txt(1696, 390, "metrics", 12, 700, ORANGE_STROKE, "end"),
    ]

    parts.append("</svg>")
    render("\n".join(parts), "architecture", W, H)


def loop_arch() -> None:
    """Inference-loop diagram — same grammar as KAITO ragarch.png."""
    W, H = 1400, 860
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs(),
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
    ]

    # User prompt
    parts += [
        rect(500, 28, 400, 72, WHITE, "#C0392B", 2.4, 16, "7 5"),
        txt(700, 58, "User prompt P", 20, 800, "#C0392B"),
        txt(700, 82, "can be millions of tokens", 13.5, 600, MUTED),
        line(700, 100, 700, 132, INK, 2.2, "arr"),
    ]

    # REPL (green, large left)
    parts += [
        rect(48, 144, 620, 420, GREEN, GREEN_STROKE, 2.6, 22),
        badge(80, 176, WHITE, GREEN_STROKE, "P"),
        txt(358, 182, "Python REPL", 24, 800, GREEN_INK),
    ]
    items = [
        ("1", "Store P as a string variable"),
        ("2", "Register sub_RLM(query, slice)"),
        ("3", "Execute generated Python"),
        ("4", "Trim stdout to metadata"),
        ("5", "Set Final when the answer is ready"),
    ]
    for i, (n, label) in enumerate(items):
        y = 230 + i * 56
        parts += [
            f'<circle cx="96" cy="{y}" r="14" fill="{WHITE}" stroke="{GREEN_STROKE}" stroke-width="1.8"/>',
            txt(96, y + 5, n, 13, 800, GREEN_STROKE),
            txt(126, y + 5, label, 16, 600, GREEN_INK, "start"),
        ]
    parts.append(txt(358, 530, "Root LM never sees raw prompt tokens", 13.5, 700, GREEN_STROKE))

    # LM (blue, large right)
    parts += [
        rect(732, 144, 620, 420, "#D6EFFF", BLUE_STROKE, 2.6, 22),
        badge(764, 176, WHITE, BLUE_STROKE, "L"),
        txt(1042, 182, "PyTorch  ·  Language Model", 22, 800, "#044868"),
    ]
    items2 = [
        ("1", "model.generate(hist)"),
        ("2", "Qwen3-8B  /  RLM-Qwen3-8B"),
        ("3", "H100 GPU  ·  vLLM"),
        ("4", "Returns Python code blocks"),
        ("5", "Never sees the full prompt P"),
    ]
    for i, (n, label) in enumerate(items2):
        y = 230 + i * 56
        parts += [
            f'<circle cx="780" cy="{y}" r="14" fill="{WHITE}" stroke="{BLUE_STROKE}" stroke-width="1.8"/>',
            txt(780, y + 5, n, 13, 800, BLUE_STROKE),
            txt(810, y + 5, label, 16, 600, "#044868", "start"),
        ]
    parts.append(txt(1042, 530, "Each call = one autoregressive pass", 13.5, 700, BLUE_STROKE))

    # Cross arrows + LOOP badge
    parts += [
        path("M 668 280 H 720", INK, 2.4, "arr"),
        path("M 720 428 H 668", INK, 2.4, "arr"),
        txt(694, 264, "metadata", 12, 800, INK),
        txt(694, 454, "code", 12, 800, INK),
        f'<rect x="668" y="338" width="64" height="26" rx="13" fill="{PINK_STROKE}"/>',
        txt(700, 356, "LOOP", 11, 800, WHITE),
    ]

    # Decision + output
    parts += [
        line(358, 564, 358, 596, INK, 2.2, "arr"),
        rect(198, 608, 320, 56, WHITE, PINK_STROKE, 2.4, 28),
        txt(358, 644, "Final set in REPL?", 16, 800, PINK_INK),
        path("M 518 636 H 620", GREEN_STROKE, 2.4, "arr-g"),
        txt(560, 624, "Yes", 12, 800, GREEN_STROKE),
        rect(632, 608, 200, 56, GREEN, GREEN_STROKE, 2.4, 14),
        txt(732, 644, "OUTPUT", 18, 800, GREEN_INK),
        path("M 198 636 H 72 V 164 H 40", "#C0392B", 2.2, "arr", "6 4"),
        txt(60, 400, "No", 12, 800, "#C0392B"),
        txt(60, 416, "repeat", 12, 800, "#C0392B"),
        rect(900, 600, 452, 80, ORANGE, ORANGE_STROKE, 2.2, 14, "5 4"),
        txt(1126, 634, "sub_RLM() recursive calls", 16, 800, ORANGE_INK),
        txt(1126, 660, "REPL code can spawn GPU passes in a loop", 13, 600, ORANGE_INK),
        path("M 668 500 Q 780 580 900 640", ORANGE_STROKE, 1.8, "arr-o", "5 4"),
    ]

    # Caption bar
    parts += [
        rect(48, 732, 1304, 88, WHITE, GROUP_STROKE, 2.0, 16),
        txt(700, 770, "Recursion is program control flow in the REPL, not extra transformer layers.", 16, 700),
        txt(700, 798, "PyTorch / vLLM runs one generate() per root step and per sub-call.", 14, 600, MUTED),
    ]

    parts.append("</svg>")
    render("\n".join(parts), "architecture-loop", W, H)


def main() -> None:
    system_arch()
    loop_arch()


if __name__ == "__main__":
    main()
