# H100 benchmark harness

This directory supplies the conference measurement layer on top of [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm). Start with the [root README](../README.md) for the research context and architecture.

## Before interpreting results

The checked-in code expresses an intended comparison. These implementation limits need to be resolved or controlled before attributing any speedup to caching or concurrency:

1. **Cache state is not verified.** `CONFIGS[*]["prefix_caching"]` is descriptive data; the runner does not send it to vLLM. The launcher passes `--enable-prefix-caching` when requested, but does not pass a disable flag otherwise. Check the installed engine's effective setting. Current vLLM exposes [`--no-enable-prefix-caching`](https://docs.vllm.ai/en/latest/cli/serve/#--enable-prefix-caching); an explicitly disabled baseline needs that setting in the server launch command for a compatible version.
2. **The concurrency preset does not prove a concurrency change.** The runner sets `max_depth=1`. In [upstream RLM commit d04208a](https://github.com/alexzhang13/rlm/blob/d04208afbad29ca675ab13478c40ee8bebc84bfe/rlm/environments/local_repl.py), `max_concurrent_subcalls` bounds the recursive callback's thread pool. With no recursive callback at depth 1, `rlm_query_batched` falls back to the plain `llm_query_batched` path, which does not use that pool limit. Confirm the installed implementation and actual request overlap before comparing the 1/4 presets. Changing recursion depth would change the workload as well.
3. **Versions and sampling are not pinned.** Requirements specify minimum versions. The runner does not set a generation seed, sampling parameters, or a dataset revision. It filters and sorts samples rather than drawing a seeded random sample. Save the package versions, model/dataset revisions, selected sample IDs, and effective server configuration alongside a reported run.
4. **Warm-up and cache state are uncontrolled.** The harness has no explicit warm-up, repeated-trial schedule, or cache reset. Reusing a server between presets can carry prefix state forward. Restart or reset consistently and state whether a comparison uses warm or cold caches.
5. **Counts and aggregates require inspection.** Missing trajectory metadata becomes zero iterations/sub-calls. Aggregates exclude samples with errors. Keep failure counts and per-sample logs alongside score and latency; a successful subset alone can be misleading.

No GPU run is implied by these documentation checks. This repository does not contain committed result JSON files.

## Set up and smoke-test

Target: one H100 80GB, CUDA drivers, Python 3.11+, `git`, `curl`, and `lsof`.

```bash
cd benchmark  # from the repository root
bash setup_h100.sh
source .venv/bin/activate

bash serve_model.sh --model Qwen/Qwen3-8B --prefix-caching
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache --samples 1
```

Setup creates a virtual environment, installs `requirements.txt`, and downloads both models. The launcher kills the process on its selected port before starting vLLM; its log is `results/vllm_server.log`. The RLM executes generated Python in the local environment.

## Run the intended comparison

After checking the conditions above, this is the existing preset sequence. The first launch still relies on the installed server's cache default unless its command is amended to explicitly disable caching.

```bash
# Run from benchmark/ with the environment activated.
for model in Qwen/Qwen3-8B mit-oasys/rlm-qwen3-8b-v0.1; do
  bash serve_model.sh --model "$model"
  python run_benchmark.py --model "$model" --config baseline --samples 20

  bash serve_model.sh --model "$model" --prefix-caching
  python run_benchmark.py --model "$model" --config prefix-cache --samples 20

  # Restart to avoid carrying the preceding preset's prefix cache into this run.
  bash serve_model.sh --model "$model" --prefix-caching
  python run_benchmark.py --model "$model" --config prefix-cache-batched --samples 20
done

python plot_results.py
```

Runs use the OOLONG-synth `test` split with `1024 < context_len <= --max-context-len`, optionally filtered by dataset name, sorted by `context_window_id`, then capped by `--samples`. Samples run sequentially. The loader passes context text plus question as `prompt`; the question is also supplied as `root_prompt`.

## CLI reference

### `run_benchmark.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | Hugging Face checkpoint ID matching the server |
| `--config` | Required | `baseline`, `prefix-cache`, or `prefix-cache-batched` |
| `--samples` | `20` | Cap after filtering and sorting |
| `--vllm-url` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `--gpu-device` | `0` | Device index for memory polling |
| `--task-filter` | Unset | Substring match on the dataset field |
| `--max-context-len` | `131072` | Upper bound on dataset context length; not the server window |
| `--output-dir` | `results` | Result JSON and RLM log directory |

### `serve_model.sh`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | Checkpoint to load |
| `--port` | `8000` | Server port |
| `--prefix-caching` | Omitted | Adds `--enable-prefix-caching`; omission does not explicitly disable it |
| `--gpu-mem` | `0.90` | GPU memory utilization target |
| `--dtype` | `bfloat16` | Model data type |

The launcher fixes `--max-model-len` at `32768`. It waits up to 300 seconds for `/health`.

### `plot_results.py`

Both `--results-dir` and `--output-dir` default to `results`. The script reads files matching `*__*.json`.

## Outputs

Each run writes `<model-name>__<config>.json`, with per-sample responses (truncated to 500 characters in the saved record), scores, timings, memory, token/trajectory counts, errors, and aggregates. Repeating the same model/config in the same directory overwrites its JSON; use a separate `--output-dir` for each trial. RLM logs go under `<output-dir>/rlm_logs/`.

| Chart | File |
| --- | --- |
| Mean wall-clock time | `chart_wall_clock.png` |
| Maximum sampled GPU memory | `chart_peak_vram.png` |
| Mean task score | `chart_accuracy.png` |
| Relative latency versus baseline | `chart_speedup.png` |
| Mean iterations and sub-calls | `chart_iterations_subcalls.png` |

Chart names retain the original preset labels. They do not validate cache state, concurrency, or experimental comparability.
