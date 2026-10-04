#!/usr/bin/env python3
"""Generate the README's research and implementation figures.

SVG needs only Python's standard library. Add --png with CairoSVG installed to
also regenerate the 2x PNG fallbacks. See architecture.md for sources and scope.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK = '#202b33'
MUTED = '#56646f'
RULE = '#c7cfd4'
BLUE = '#285e83'
BLUE_BG = '#eef5fa'
GREEN = '#326754'
GREEN_BG = '#eff6f1'
FONT = 'Arial, Helvetica, sans-serif'


def text(x, y, value, size=17, color=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')


def box(x, y, w, h, fill='#fff', stroke=RULE):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def arrow(d, dashed=False, color=INK):
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'marker-end="url(#arrow)"{dash}/>')


def rule(x1, y, x2):
    return f'<path d="M{x1} {y} H{x2}" stroke="{RULE}" stroke-width="1"/>'


def figure(height, title, description, content):
    return '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#202b33"/></marker></defs>',
        f'<rect width="960" height="{height}" fill="#ffffff"/>',
        *content, '</svg>',
    ]) + '\n'


def inference_loop():
    s = [
        text(32, 35, '01 / RLM INFERENCE', 13, MUTED, 700),
        text(32, 70, 'Long input lives in the program’s state', 28, weight=700),
        text(32, 99, 'Conceptual loop from Zhang et al., §2 / Algorithm 1. Serving details are in Figure 2.', 16, MUTED),
        rule(32, 120, 928),
        text(688, 159, 'Long input P', 19, weight=700, anchor='middle'),
        arrow('M688 170 V219'),
        text(704, 200, 'load once', 15, MUTED),
        box(42, 220, 294, 168, BLUE_BG, BLUE),
        text(62, 251, 'Root language model', 22, BLUE, 700),
        text(62, 281, 'Task + input metadata', 18),
        text(62, 307, 'Code and feedback history', 18),
        text(62, 359, 'Each call has a finite context window.', 15, MUTED),
        box(548, 220, 380, 168, GREEN_BG, GREEN),
        text(568, 251, 'Persistent Python REPL', 22, GREEN, 700),
        text(568, 281, 'P · selected slices · intermediate values', 18),
        text(568, 307, 'Execute code; retain state across turns', 18),
        text(568, 359, 'Full input is available through variables.', 15, MUTED),
        arrow('M336 275 H548'),
        text(442, 262, 'generated code', 16, anchor='middle'),
        arrow('M548 330 H336'),
        text(442, 319, 'bounded feedback', 16, anchor='middle'),
        # Initial metadata takes a separate path from ongoing feedback.
        arrow('M585 220 V183 H189 V220'),
        text(375, 172, 'initial input metadata', 15, MUTED, anchor='middle'),
        box(548, 466, 380, 107, BLUE_BG, BLUE),
        text(568, 498, 'Sub-call on a constructed prompt', 21, BLUE, 700),
        text(568, 526, 'Selected text or a transformed subproblem', 17),
        text(568, 551, 'May recurse if the runtime permits it', 16, MUTED),
        arrow('M614 388 V466'),
        text(602, 424, 'prompt', 15, MUTED, anchor='end'),
        arrow('M858 466 V388'),
        text(871, 438, 'result', 15, MUTED),
        # A final answer is read from state, not forced through the root output.
        arrow('M548 367 H505 V518 H336'),
        text(491, 483, 'finish', 15, MUTED, anchor='end'),
        box(42, 485, 294, 68),
        text(62, 513, 'Return final answer', 20, weight=700),
        text(62, 538, 'Read the completed result from state', 15, MUTED),
        rule(32, 605, 928),
        text(32, 635, 'The full input stays outside the root prompt; selected excerpts can still enter model calls.', 17),
        text(32, 661, 'History can grow across turns. This is programmatic decomposition, not an unlimited model window.', 16, MUTED),
    ]
    return figure(688, 'RLM inference loop',
                  'Long input is loaded into persistent REPL state. The root model receives metadata, generates code, and receives bounded feedback. Code can make sub-calls on selected text and retain their results. A final answer is returned from state. This is the paper’s conceptual algorithm, not a claim of unlimited recursion in this benchmark.', s)


def architecture():
    s = [
        text(32, 35, '02 / BENCHMARK IMPLEMENTATION', 13, MUTED, 700),
        text(32, 70, 'One Python harness, one model server', 28, weight=700),
        text(32, 99, 'Root and sub-call requests use the same configured endpoint and checkpoint.', 16, MUTED),
        rule(32, 120, 928),
        text(42, 153, 'PYTHON HOST PROCESS', 13, MUTED, 700),
        text(702, 153, 'vLLM MODEL SERVER', 13, MUTED, 700),
        box(32, 170, 602, 281, '#fafbfc'),
        box(54, 194, 230, 68),
        text(70, 220, 'OOLONG-synth', 20, weight=700),
        text(70, 244, 'Filtered, sorted test samples', 15, MUTED),
        arrow('M169 262 V313'),
        text(184, 292, 'prompt + question', 15, MUTED),
        box(54, 313, 230, 109),
        text(70, 342, 'Benchmark runner', 20, weight=700),
        text(70, 370, 'One sample at a time', 17),
        text(70, 398, 'run_benchmark.py', 15, MUTED),
        box(356, 194, 256, 228, GREEN_BG, GREEN),
        text(374, 226, 'Upstream RLM runtime', 20, GREEN, 700),
        text(374, 255, 'Root history + local REPL', 17),
        text(374, 281, 'Constructs model requests', 17),
        rule(374, 302, 594),
        text(374, 329, 'max_iterations = 30', 16),
        text(374, 355, 'max_depth = 1', 16),
        text(374, 395, 'Sub-calls are leaf LM calls.*', 15, MUTED),
        arrow('M284 335 H356'),
        text(320, 323, 'call', 14, MUTED, anchor='middle'),
        arrow('M356 390 H284'),
        text(320, 381, 'return', 14, MUTED, anchor='middle'),
        box(702, 170, 226, 281, BLUE_BG, BLUE),
        text(720, 204, 'OpenAI-compatible API', 18, BLUE, 700),
        text(720, 235, 'vLLM scheduler', 18),
        text(720, 263, 'Model weights + KV cache', 16),
        text(720, 299, 'PyTorch / CUDA execution', 16),
        rule(720, 322, 910),
        text(720, 351, '1 × NVIDIA H100 80 GB', 17, weight=700),
        text(720, 378, 'bfloat16 · 32,768 tokens', 16),
        text(720, 405, 'One checkpoint per run', 15, MUTED),
        arrow('M612 239 H702'),
        text(657, 224, 'requests', 14, MUTED, anchor='middle'),
        arrow('M702 286 H612'),
        text(657, 310, 'responses', 14, MUTED, anchor='middle'),
        # Observation is distinguished from the inference path by dashes.
        box(54, 524, 558, 95),
        text(72, 554, 'Score and record each sample', 21, weight=700),
        text(72, 582, 'Answer score · elapsed time · usage / trajectory · GPU memory', 16),
        text(72, 605, 'JSON results → plot_results.py → five charts', 15, MUTED),
        arrow('M169 422 V524', True),
        text(182, 488, 'completion + timing', 15, MUTED),
        arrow('M815 451 V567 H612', True),
        text(724, 551, 'VRAM poll / 0.5 s', 15, MUTED, anchor='middle'),
        rule(32, 650, 928),
        text(32, 680, '* In the upstream implementation reviewed; dependency versions are not pinned here.', 16, MUTED),
        text(32, 706, 'Solid arrows: calls and returned data. Dashed arrows: measurements collected by this repository.', 16, MUTED),
    ]
    return figure(733, 'Benchmark implementation and measurement boundaries',
                  'OOLONG samples enter a sequential Python benchmark runner. It calls the upstream RLM runtime with a local REPL, 30 root iterations and max depth 1. Root and leaf sub-call requests share a vLLM endpoint running one checkpoint on one H100. The harness records scores, timing, usage and trajectory data, and polls device memory. Dashed arrows show measurement paths.', s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true', help='Also render 2x PNGs (requires CairoSVG)')
    args = parser.parse_args()
    for name, svg in [('architecture-loop', inference_loop()), ('architecture', architecture())]:
        target = OUT / f'{name}.svg'
        target.write_text(svg, encoding='utf-8')
        print(f'Wrote {target.name}')
        if args.png:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / f'{name}.png'), scale=2)
            print(f'Wrote {name}.png')


if __name__ == '__main__':
    main()
