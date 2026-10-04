# Architecture: sources and figure notes

The figures are for readers who know language-model inference but are new to RLMs. They explain where long input is stored, how model calls interact with program state, and what this repository measures. Figure 1 describes the research algorithm; Figure 2 describes the checked-in benchmark. Neither represents a measured request trace or a deployment with multiple GPUs.

## Evidence

Reviewed against repository base `63eab4e`, the paper's v2, and upstream RLM commit [`d04208a`](https://github.com/alexzhang13/rlm/tree/d04208afbad29ca675ab13478c40ee8bebc84bfe). That upstream snapshot is a documentation reference, not a dependency pin. `benchmark/requirements.txt` has lower bounds, so an installed release can behave differently.

| Claim | Status | Evidence | Consequence for the figures |
| --- | --- | --- | --- |
| Long input is stored in an external REPL; code manipulates it and retains intermediate values. | Confirmed research design | [Paper §2, Algorithm 1](https://arxiv.org/html/2512.24601v2#S2) | Input arrow enters the REPL, not the root LM. |
| Metadata and feedback can be bounded per step; the root history still accumulates. | Confirmed research design | [Paper §2](https://arxiv.org/html/2512.24601v2#S2) | Feedback arrow says “bounded”; the model box retains a finite context window. |
| Code may invoke a sub-RLM and read a final value from state. | Confirmed research design | [Paper Algorithm 1](https://arxiv.org/html/2512.24601v2#S2) | Sub-call returns to REPL state; final answer exits from that state. |
| This runner uses local execution, 30 iterations, and depth 1. | Confirmed repository configuration | [`run_benchmark.py`](../benchmark/run_benchmark.py), `run_benchmark()` | Figure 2 labels exact configured limits. |
| Local REPL code runs in the RLM host process. | Confirmed in reviewed upstream revision | [`LocalREPL`](https://github.com/alexzhang13/rlm/blob/d04208afbad29ca675ab13478c40ee8bebc84bfe/rlm/environments/local_repl.py) | Runner and runtime are contained in one Python boundary. |
| At depth 1, sub-calls use plain LM completions. | Depends on installed runtime; confirmed in reviewed snapshot | [`RLM._spawn_completion_context`](https://github.com/alexzhang13/rlm/blob/d04208afbad29ca675ab13478c40ee8bebc84bfe/rlm/core/rlm.py) and [`LocalREPL._rlm_query`](https://github.com/alexzhang13/rlm/blob/d04208afbad29ca675ab13478c40ee8bebc84bfe/rlm/environments/local_repl.py) | Figure 2 does not draw recursive child REPLs. |
| Root and sub-calls share the configured endpoint/checkpoint. | Repository configuration interpreted through reviewed runtime | Runner supplies one `backend_kwargs` and no alternate backend | One server, with request and response arrows. |
| Server limit is 32,768 tokens; input filter permits up to 131,072. | Confirmed repository configuration | [`serve_model.sh`](../benchmark/serve_model.sh), [`oolong_loader.py`](../benchmark/tasks/oolong_loader.py) | GPU/model context limit is distinct from REPL input size. |
| VRAM is polled device-wide every 0.5 seconds. | Confirmed measurement method | [`collector.py`](../benchmark/metrics/collector.py), runner | Dashed measurement arrow from the GPU/server group. |
| Prefix caching reuses matching KV prefixes and saves prefill, not decode. | Confirmed serving mechanism | [vLLM APC documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Caching belongs inside the server; it is not a separate service. |
| The 1-versus-4 setting changes realized concurrency. | **Not established** | [`LocalREPL._rlm_query_batched`](https://github.com/alexzhang13/rlm/blob/d04208afbad29ca675ab13478c40ee8bebc84bfe/rlm/environments/local_repl.py) uses the limit only with a recursive callback; depth 1 falls back to the plain batched LM path | No four-worker pool or implied speedup is drawn. |

The performance hypotheses in the README are hypotheses, not observed results. No timings or performance ratios are encoded in arrow lengths or box sizes.

## Reading the figures

Blue identifies language-model inference; pale green identifies the Python environment and its state. Neutral boxes identify dataset handling, harness code, and reporting. Color is redundant with the labels.

Figure 1 separates the root model's history from the REPL's state. The initial metadata path and the iterative feedback path are distinct. The generic sub-call box permits recursion as a property of the research design without prescribing an API function name. The paper's `sub_RLM` / `Final` notation and the library's `llm_query`, `rlm_query`, and final-answer parsing are not interchangeable API contracts.

Figure 2 shows software/process boundaries, not a distributed cluster. Its vLLM box groups API handling, scheduling, PyTorch/CUDA execution, and the target GPU. The same checkpoint serves root and leaf calls. A single bidirectional request path is drawn as two labeled arrows; it does not imply two separate model instances. The collector's dashed paths show observations rather than inference requests.

The local REPL state is persistent during a completion's reasoning loop. The diagram does not imply state reuse between different benchmark samples. The full input is accessible outside the root window, but snippets selected by the program can still be shown to the model.

## Regenerate

From the repository root, regenerate SVG using only the Python standard library:

```bash
python3 docs/generate_architecture.py
```

To also render the checked-in 2× PNG exports, install CairoSVG and its Cairo system library, then run:

```bash
python3 docs/generate_architecture.py --png
```

Both SVGs have a white canvas, embedded text, accessible titles and descriptions, and no external fonts or images. The README embeds SVG for sharp rendering at different widths; PNGs remain available for slides and other tools. Inspect both figures after editing, especially edge labels, return paths, and the distinction between the conceptual loop and configured depth.
