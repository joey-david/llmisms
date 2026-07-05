from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np


def entropy_and_surprisal(
    logits: np.ndarray, selected_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact full-vocabulary Shannon entropy and selected-token surprisal."""
    values = np.asarray(logits, dtype=np.float32)
    ids = np.asarray(selected_ids, dtype=np.int64)
    if values.ndim != 2 or ids.shape != (values.shape[0],):
        raise ValueError("Expected logits [tokens, vocabulary] and one id per token")
    maxima = values.max(axis=-1, keepdims=True)
    shifted = values - maxima
    log_z = np.log(np.exp(shifted).sum(axis=-1, dtype=np.float64)).astype(
        np.float32
    )
    probabilities = np.exp(shifted - log_z[:, None]).astype(np.float32)
    entropy = log_z - (probabilities * shifted).sum(axis=-1, dtype=np.float64)
    surprisal = log_z - shifted[np.arange(len(ids)), ids]
    return entropy.astype(np.float32), surprisal.astype(np.float32)


def derived_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join((str(base_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def conditioning_gain(nll_with: float, nll_deleted: float) -> float:
    """Decrease in continuation NLL caused by retaining a preceding span."""
    return float(nll_deleted - nll_with)


def stage_deltas(values: dict[str, float]) -> tuple[float, float]:
    return values["dpo"] - values["sft"], values["rlvr"] - values["dpo"]


def validate_stage_pairing(rows: Sequence[dict]) -> dict[str, int | bool | str | None]:
    """Verify one stochastic response per prompt/seed at every study stage."""
    stages = {"base", "sft", "dpo", "rlvr"}
    pairs: dict[tuple[str, int], list[str]] = {}
    duplicate_count = 0
    for row in rows:
        if row.get("greedy", False) or row.get("stage") not in stages:
            continue
        key = (str(row["prompt_id"]), int(row["base_seed"]))
        present = pairs.setdefault(key, [])
        if row["stage"] in present:
            duplicate_count += 1
        present.append(row["stage"])
    incomplete_count = sum(set(present) != stages for present in pairs.values())
    if not pairs:
        return {
            "pair_count": 0,
            "incomplete_pair_count": 0,
            "duplicate_count": duplicate_count,
            "valid": None,
            "status": "not_applicable",
        }
    return {
        "pair_count": len(pairs),
        "incomplete_pair_count": incomplete_count,
        "duplicate_count": duplicate_count,
        "valid": incomplete_count == 0 and duplicate_count == 0,
        "status": "complete" if incomplete_count == 0 else "incomplete",
    }


def clustered_bootstrap(
    rows: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    cluster_key: str = "prompt_id",
    iterations: int = 2000,
    seed: int = 202707,
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[cluster_key]), []).append(row)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sample = [row for key in sampled for row in grouped[str(key)]]
        estimates[index] = statistic(sample)
    estimate = float(statistic(rows))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted
