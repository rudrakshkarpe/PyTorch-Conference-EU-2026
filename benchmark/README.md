# H100 benchmark harness

This directory is the conference measurement layer on top of [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm). Architecture, what we added, and the full CLI live in the [root README](../README.md).

## Quick start

```bash
bash setup_h100.sh
source .venv/bin/activate

bash serve_model.sh --model Qwen/Qwen3-8B
python run_benchmark.py --model Qwen/Qwen3-8B --config baseline --samples 20

bash serve_model.sh --model Qwen/Qwen3-8B --prefix-caching
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache --samples 20
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache-batched --samples 20

python plot_results.py
```

Repeat the same three configs for `mit-oasys/rlm-qwen3-8b-v0.1`. Outputs go to `results/`.
