"""Load OOLONG-synth benchmark samples from HuggingFace and score responses."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import dateutil.parser
from datasets import load_dataset


@dataclass
class OolongSample:
    id: str
    context_window_id: str
    dataset_name: str
    task: str
    answer_type: str
    prompt: str
    question: str
    gold_answer: str
    context_len: int


def load_oolong_synth(
    max_samples: int | None = None,
    max_context_len: int = 131072,
    min_context_len: int = 1024,
    task_filter: str | None = None,
) -> list[OolongSample]:
    """Download oolong-synth from HF and return formatted samples.

    Args:
        max_samples: Cap the number of samples returned.
        max_context_len: Upper bound on context token estimate.
        min_context_len: Lower bound on context token estimate.
        task_filter: If set, only include samples whose ``dataset`` field
            contains this substring (e.g. ``"trec_coarse"``).
    """
    ds = load_dataset("oolongbench/oolong-synth", split="test")

    if task_filter:
        ds = ds.filter(lambda x: task_filter in x.get("dataset", ""))

    ds = ds.filter(lambda x: min_context_len < x["context_len"] <= max_context_len)
    ds = ds.sort("context_window_id")

    samples: list[OolongSample] = []
    for row in ds:
        full_prompt = row["context_window_text"] + "\n" + row["question"]
        samples.append(
            OolongSample(
                id=row["id"],
                context_window_id=row["context_window_id"],
                dataset_name=row.get("dataset", ""),
                task=row.get("task", ""),
                answer_type=row.get("answer_type", ""),
                prompt=full_prompt,
                question=row["question"],
                gold_answer=str(row["answer"]).strip(),
                context_len=row["context_len"],
            )
        )

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


# ---------------------------------------------------------------------------
# Scoring (mirrors the official eval_helpers.py from abertsch72/oolong)
# ---------------------------------------------------------------------------

def _attempt_parse(answer: str) -> tuple[str, str]:
    """Extract the model's answer from its response text."""
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence

    candidate = answer.split(":")[-1].strip()
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")
    parse_confidence = "med"

    if any(k in answer for k in ("User:", "Answer:", "Date:", "Label")):
        parse_confidence = "high"
    if len(candidate) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"

    return candidate, parse_confidence


def score_response(sample: OolongSample, model_output: str) -> dict[str, Any]:
    """Score a single model response against the gold answer.

    Returns a dict with ``score`` (float 0-1), ``parsed_answer``, and
    ``parse_confidence``.
    """
    raw_gold = sample.gold_answer
    if "datetime" in raw_gold:
        gold = datetime.strptime(raw_gold, "[datetime.date(%Y, %m, %d)]")
    else:
        try:
            gold = ast.literal_eval(raw_gold)
            if isinstance(gold, list) and len(gold) == 1:
                gold = gold[0]
        except (ValueError, SyntaxError):
            gold = raw_gold

    parsed, confidence = _attempt_parse(model_output)
    score = 0.0

    if str(parsed) == str(gold):
        score = 1.0
    elif str(parsed) in ("more common", "less common", "same frequency"):
        if str(parsed) in str(gold):
            score = 1.0
    elif sample.answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            score = 0.75 ** abs(int(gold) - int(parsed))
        except (ValueError, TypeError):
            confidence = "low"
    elif sample.answer_type == "ANSWER_TYPE.DATE":
        try:
            parsed_dt = dateutil.parser.parse(str(parsed))
            score = float(parsed_dt == gold)
        except (ValueError, TypeError):
            confidence = "low"

    return {
        "score": score,
        "parsed_answer": str(parsed),
        "gold_answer": str(gold),
        "parse_confidence": confidence,
    }
