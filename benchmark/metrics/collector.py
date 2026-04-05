"""Metrics collection utilities: VRAM monitoring, wall-clock timing, RLM metadata extraction."""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# VRAM Monitor  (background thread polling nvidia-smi)
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """Poll GPU VRAM usage in a background thread and track the peak.

    Uses ``nvidia-smi`` so it works without importing ``pynvml`` (though
    ``nvidia-ml-py3`` is preferred when available).
    """

    def __init__(self, device_index: int = 0, interval: float = 0.5):
        self.device_index = device_index
        self.interval = interval
        self._peak_mib: float = 0.0
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- nvidia-smi fallback --------------------------------------------------

    @staticmethod
    def _query_vram_nvidia_smi(device_index: int) -> float:
        """Return current VRAM usage in MiB via nvidia-smi."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            return float(out.strip())
        except Exception:
            return 0.0

    # -- pynvml (preferred) ---------------------------------------------------

    @staticmethod
    def _query_vram_pynvml(handle) -> float:
        import pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)  # bytes -> MiB

    # -- polling loop ---------------------------------------------------------

    def _poll(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            query = lambda: self._query_vram_pynvml(handle)
        except Exception:
            query = lambda: self._query_vram_nvidia_smi(self.device_index)

        while not self._stop.is_set():
            used = query()
            self._samples.append(used)
            if used > self._peak_mib:
                self._peak_mib = used
            self._stop.wait(self.interval)

    def start(self):
        self._peak_mib = 0.0
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak VRAM in MiB."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak_mib

    @property
    def peak_mib(self) -> float:
        return self._peak_mib

    @property
    def samples(self) -> list[float]:
        return list(self._samples)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer():
    """Yield a dict that will contain ``elapsed_s`` after the block exits."""
    result: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_s"] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# RLM metadata extraction
# ---------------------------------------------------------------------------

@dataclass
class RLMMetrics:
    """Structured metrics extracted from a single RLM completion."""

    wall_clock_s: float = 0.0
    peak_vram_mib: float = 0.0
    vram_samples: list[float] = field(default_factory=list)
    task_score: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    subcall_count: int = 0
    model_response: str = ""
    error: str | None = None


def extract_rlm_metrics(completion: Any) -> dict[str, Any]:
    """Pull iteration count, sub-call count, and token usage from an
    ``RLMChatCompletion`` object returned by ``rlm.completion()``.

    Fields on RLMChatCompletion:
      - response: str
      - usage_summary: UsageSummary (has .model_usage_summaries dict)
      - execution_time: float (seconds)
      - metadata: dict | None (full trajectory when RLMLogger is used)
    """
    result: dict[str, Any] = {
        "iterations": 0,
        "subcall_count": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model_response": "",
        "execution_time": 0.0,
    }

    if completion is None:
        return result

    result["model_response"] = getattr(completion, "response", "") or ""
    result["execution_time"] = getattr(completion, "execution_time", 0.0)

    # Token usage from usage_summary
    usage_summary = getattr(completion, "usage_summary", None)
    if usage_summary and hasattr(usage_summary, "model_usage_summaries"):
        for _model, summary in usage_summary.model_usage_summaries.items():
            result["input_tokens"] += getattr(summary, "total_input_tokens", 0)
            result["output_tokens"] += getattr(summary, "total_output_tokens", 0)
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]

    # Iteration + sub-call counts from metadata (populated by RLMLogger)
    metadata = getattr(completion, "metadata", None)
    if metadata and isinstance(metadata, dict):
        iters = metadata.get("iterations", [])
        result["iterations"] = len(iters)
        total_subcalls = 0
        for it in iters:
            code_blocks = it.get("code_blocks", [])
            for cb in code_blocks:
                rlm_calls = cb.get("result", {}).get("rlm_calls", [])
                total_subcalls += len(rlm_calls)
        result["subcall_count"] = total_subcalls

    return result
