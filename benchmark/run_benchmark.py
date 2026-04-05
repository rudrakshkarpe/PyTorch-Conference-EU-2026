#!/usr/bin/env python3
"""RLM Inference Benchmark Runner.

Runs OOLONG-synth samples through the RLM library with a locally-served vLLM
model, collecting task scores, timing, VRAM, token counts, and RLM-specific
metrics (iterations, sub-calls).

Usage:
    python run_benchmark.py --model Qwen/Qwen3-8B --config baseline --samples 20
    python run_benchmark.py --model mit-oasys/rlm-qwen3-8b-v0.1 --config prefix-cache-batched
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rlm import RLM
from rlm.logger import RLMLogger

from metrics.collector import RLMMetrics, VRAMMonitor, extract_rlm_metrics, timer
from tasks.oolong_loader import OolongSample, load_oolong_synth, score_response

# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline": {
        "description": "vLLM defaults, sequential sub-calls",
        "prefix_caching": False,
        "max_concurrent_subcalls": 1,
    },
    "prefix-cache": {
        "description": "vLLM prefix caching ON, sequential sub-calls",
        "prefix_caching": True,
        "max_concurrent_subcalls": 1,
    },
    "prefix-cache-batched": {
        "description": "vLLM prefix caching ON, concurrent sub-calls (4)",
        "prefix_caching": True,
        "max_concurrent_subcalls": 4,
    },
}


def make_safe_name(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def run_single_sample(
    rlm_client: RLM,
    sample: OolongSample,
    vram_monitor: VRAMMonitor,
) -> RLMMetrics:
    """Run one OOLONG sample and collect all metrics."""
    metrics = RLMMetrics()

    vram_monitor.start()
    with timer() as t:
        try:
            completion = rlm_client.completion(
                sample.prompt,
                root_prompt=sample.question,
            )
            rlm_data = extract_rlm_metrics(completion)
            metrics.model_response = rlm_data["model_response"]
            metrics.iterations = rlm_data["iterations"]
            metrics.subcall_count = rlm_data["subcall_count"]
            metrics.total_tokens = rlm_data["total_tokens"]
            metrics.input_tokens = rlm_data["input_tokens"]
            metrics.output_tokens = rlm_data["output_tokens"]
        except Exception as exc:
            metrics.error = str(exc)
            print(f"  ERROR: {exc}")

    metrics.wall_clock_s = t["elapsed_s"]
    metrics.peak_vram_mib = vram_monitor.stop()
    metrics.vram_samples = vram_monitor.samples

    if metrics.model_response and not metrics.error:
        scoring = score_response(sample, metrics.model_response)
        metrics.task_score = scoring["score"]

    return metrics


def run_benchmark(
    model: str,
    config_name: str,
    vllm_base_url: str,
    samples: list[OolongSample],
    gpu_device: int,
    log_dir: str | None,
) -> list[dict]:
    """Run full benchmark: all samples through one (model, config) combo."""
    cfg = CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"Model:  {model}")
    print(f"Config: {config_name} - {cfg['description']}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}\n")

    logger = RLMLogger(log_dir=log_dir) if log_dir else RLMLogger()

    rlm_client = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model,
            "base_url": vllm_base_url,
        },
        environment="local",
        max_iterations=30,
        max_depth=1,
        max_concurrent_subcalls=cfg["max_concurrent_subcalls"],
        logger=logger,
        verbose=False,
    )

    vram_monitor = VRAMMonitor(device_index=gpu_device, interval=0.5)
    results = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] id={sample.id}  context_len={sample.context_len} ...", end=" ", flush=True)

        metrics = run_single_sample(rlm_client, sample, vram_monitor)

        result = {
            "sample_id": sample.id,
            "context_window_id": sample.context_window_id,
            "dataset": sample.dataset_name,
            "task": sample.task,
            "context_len": sample.context_len,
            "answer_type": sample.answer_type,
            "gold_answer": sample.gold_answer,
            "model_response": metrics.model_response[:500],
            "task_score": metrics.task_score,
            "wall_clock_s": round(metrics.wall_clock_s, 2),
            "peak_vram_mib": round(metrics.peak_vram_mib, 1),
            "total_tokens": metrics.total_tokens,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "iterations": metrics.iterations,
            "subcall_count": metrics.subcall_count,
            "error": metrics.error,
        }
        results.append(result)

        status = f"score={metrics.task_score:.2f}" if not metrics.error else "FAILED"
        print(
            f"{status}  "
            f"time={metrics.wall_clock_s:.1f}s  "
            f"vram={metrics.peak_vram_mib:.0f}MiB  "
            f"iters={metrics.iterations}  "
            f"subcalls={metrics.subcall_count}  "
            f"tokens={metrics.total_tokens}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="RLM Inference Benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--config",
        required=True,
        choices=list(CONFIGS.keys()),
        help="Optimization configuration",
    )
    parser.add_argument("--samples", type=int, default=20, help="Max samples to run")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index for VRAM monitoring")
    parser.add_argument("--task-filter", default=None, help="OOLONG dataset filter (e.g. trec_coarse)")
    parser.add_argument("--max-context-len", type=int, default=131072, help="Max context length")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(output_dir / "rlm_logs")

    print("Loading OOLONG-synth dataset ...")
    samples = load_oolong_synth(
        max_samples=args.samples,
        max_context_len=args.max_context_len,
        task_filter=args.task_filter,
    )
    print(f"Loaded {len(samples)} samples\n")

    if not samples:
        print("ERROR: No samples loaded. Check dataset filters.")
        sys.exit(1)

    results = run_benchmark(
        model=args.model,
        config_name=args.config,
        vllm_base_url=args.vllm_url,
        samples=samples,
        gpu_device=args.gpu_device,
        log_dir=log_dir,
    )

    # Compute aggregate stats
    valid = [r for r in results if r["error"] is None]
    agg = {}
    if valid:
        agg = {
            "mean_score": round(sum(r["task_score"] for r in valid) / len(valid), 4),
            "mean_wall_clock_s": round(sum(r["wall_clock_s"] for r in valid) / len(valid), 2),
            "max_peak_vram_mib": round(max(r["peak_vram_mib"] for r in valid), 1),
            "mean_total_tokens": round(sum(r["total_tokens"] for r in valid) / len(valid)),
            "mean_iterations": round(sum(r["iterations"] for r in valid) / len(valid), 1),
            "mean_subcalls": round(sum(r["subcall_count"] for r in valid) / len(valid), 1),
            "total_samples": len(results),
            "successful_samples": len(valid),
            "failed_samples": len(results) - len(valid),
        }

    output = {
        "model": args.model,
        "config": args.config,
        "config_description": CONFIGS[args.config]["description"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(results),
        "aggregates": agg,
        "per_sample": results,
    }

    safe_model = make_safe_name(args.model)
    out_path = output_dir / f"{safe_model}__{args.config}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
