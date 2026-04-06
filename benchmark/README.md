# RLM Inference Benchmark POC

Benchmark comparing **Qwen3-8B** (base) vs **RLM-Qwen3-8B** (SFT fine-tuned) across
three vLLM serving configurations on a single H100 80GB GPU.

## Optimization Configurations

| Config | Prefix Caching | Concurrent Sub-calls |
|--------|:-:|:-:|
| `baseline` | Off | 1 (sequential) |
| `prefix-cache` | On | 1 (sequential) |
| `prefix-cache-batched` | On | 4 (parallel) |

## Metrics Collected

- Task score (OOLONG scoring)
- Wall-clock time (seconds per query)
- Peak VRAM (nvidia-smi polling)
- Total tokens generated
- REPL iterations
- Sub-call count

## Quick Start

```bash
# 1. Provision the H100 instance
bash setup_h100.sh

# 2. Run all benchmarks (restarts vLLM between model swaps)
bash serve_model.sh --model Qwen/Qwen3-8B
python run_benchmark.py --model Qwen/Qwen3-8B --config baseline --samples 20
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache --samples 20
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache-batched --samples 20

bash serve_model.sh --model mit-oasys/rlm-qwen3-8b-v0.1
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config baseline --samples 20
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config prefix-cache --samples 20
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config prefix-cache-batched --samples 20

# 3. Generate poster-ready charts
python plot_results.py
```

Results land in `results/` as JSON files and PNG charts.
