# 🔥 RLM Inference Benchmark — PyTorch Conference EU 2026 (Paris)

> **Presentation & poster repository** for the talk *"Accelerating RLM Inference with vLLM Optimizations"* at [PyTorch Conference EU 2026](https://pytorch.org/), Paris.

---

## 📌 Overview

This repository contains the **full benchmark suite** used to produce results and charts for the conference poster. It measures the real-world inference performance of two models:

| Model | Description |
|---|---|
| [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Base model (no fine-tuning) |
| [`mit-oasys/rlm-qwen3-8b-v0.1`](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1) | SFT fine-tuned with RLM training |

Both models are evaluated across **three vLLM serving configurations** on a single **NVIDIA H100 80GB GPU** using the [OOLONG-synth](https://huggingface.co/datasets/oolongbench/oolong-synth) long-context benchmark dataset.

---

## 🗂️ Repository Structure

```
PyTorch-Conference-EU-2026/
├── benchmark/
│   ├── run_benchmark.py        # Main benchmark runner
│   ├── plot_results.py         # Chart generation (poster-ready PNGs)
│   ├── serve_model.sh          # vLLM server launcher with health check
│   ├── setup_h100.sh           # H100 environment setup & model pre-download
│   ├── requirements.txt        # Python dependencies
│   ├── metrics/
│   │   └── collector.py        # VRAM monitor, wall-clock timer, RLM metric extractor
│   ├── tasks/
│   │   └── oolong_loader.py    # OOLONG-synth dataset loader & answer scorer
│   └── results/                # Output directory (JSON results + PNG charts)
└── poster/
    ├── PyTorch Conference_2026_Paris.pdf   # Final conference poster (PDF)
    ├── pytorch-rlm-poster.html             # Interactive HTML version of the poster
    ├── pytorch-logo.png                    # PyTorch logo asset
    └── qr-code.png                         # QR code linking to this repository
```

---

## ⚙️ Optimization Configurations

Three vLLM serving configurations are benchmarked to measure the cumulative effect of each optimization:

| Config | Prefix Caching | Concurrent Sub-calls | Description |
|---|:-:|:-:|---|
| `baseline` | ❌ Off | 1 (sequential) | vLLM defaults, no optimizations |
| `prefix-cache` | ✅ On | 1 (sequential) | Automatic prefix caching enabled |
| `prefix-cache-batched` | ✅ On | 4 (parallel) | Prefix caching + parallel RLM sub-calls |

---

## 📊 Metrics Collected

For every sample, the following metrics are recorded:

| Metric | Description |
|---|---|
| **Task Score** | OOLONG-synth accuracy score (0–1), mirrors official eval |
| **Wall-Clock Time** | Seconds per query (end-to-end latency) |
| **Peak VRAM** | Maximum GPU memory used (MiB), polled via `nvidia-smi` / `pynvml` |
| **Total Tokens** | Input + output tokens consumed per query |
| **REPL Iterations** | Number of RLM reasoning iterations |
| **Sub-call Count** | Total RLM sub-calls executed per query |

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA H100 (or equivalent) GPU with CUDA drivers
- Python 3.11+
- `git`, `curl`, `lsof`

### 1. Provision the H100 instance
```bash
cd benchmark
bash setup_h100.sh
```

### 2. Activate the environment
```bash
source .venv/bin/activate
```

### 3. Run benchmarks
```bash
# Qwen3-8B (base)
bash serve_model.sh --model Qwen/Qwen3-8B
python run_benchmark.py --model Qwen/Qwen3-8B --config baseline --samples 20
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache --samples 20
python run_benchmark.py --model Qwen/Qwen3-8B --config prefix-cache-batched --samples 20

# RLM-Qwen3-8B (fine-tuned)
bash serve_model.sh --model mit-oasys/rlm-qwen3-8b-v0.1
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config baseline --samples 20
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config prefix-cache --samples 20
python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config prefix-cache-batched --samples 20
```

### 4. Generate charts
```bash
python plot_results.py
```

---

## 📈 Generated Charts

| Chart | File | Description |
|---|---|---|
| Wall-Clock Time | `chart_wall_clock.png` | Mean seconds per query per config |
| Peak VRAM Usage | `chart_peak_vram.png` | Max GPU memory (GiB) per config |
| Task Accuracy | `chart_accuracy.png` | OOLONG-synth score heatmap |
| Speedup vs Baseline | `chart_speedup.png` | Relative latency speedup factor |
| Iterations & Sub-calls | `chart_iterations_subcalls.png` | Mean REPL iterations and sub-call counts |

---

## 🛠️ CLI Reference

### `run_benchmark.py`
```
--model MODEL               HuggingFace model ID (required)
--config CONFIG             baseline | prefix-cache | prefix-cache-batched (required)
--samples N                 Number of samples (default: 20)
--vllm-url URL              vLLM server URL (default: http://localhost:8000/v1)
--gpu-device N              GPU device index (default: 0)
--task-filter STRING        Filter OOLONG dataset by task name
--max-context-len N         Max context length in tokens (default: 131072)
--output-dir DIR            Output directory (default: results/)
```

### `serve_model.sh`
```
--model <id>            HuggingFace model ID (required)
--port <int>            Server port (default: 8000)
--prefix-caching        Enable vLLM automatic prefix caching
--gpu-mem <float>       GPU memory utilization 0–1 (default: 0.90)
--dtype <str>           Model dtype (default: bfloat16)
```

---

## 📦 Dependencies

```
rlms>=0.1.1
vllm>=0.8.0
datasets>=3.0.0
matplotlib>=3.9.0
nvidia-ml-py3>=7.352.0
numpy>=1.26.0
```

---

## 📄 Poster

- 📄 **[PDF Poster](./poster/PyTorch%20Conference_2026_Paris.pdf)**
- 🌐 **[Interactive HTML Poster](./poster/pytorch-rlm-poster.html)**

---

## 👤 Author

**Rudraksh Karpe** — [@rudrakshkarpe](https://github.com/rudrakshkarpe)  
Presented at **PyTorch Conference EU 2026**, Paris 🇫🇷
