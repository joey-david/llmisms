from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import STAGE_ORDER
from .math_utils import (
    clustered_bootstrap,
    holm_adjust,
    stage_deltas,
    validate_stage_pairing,
)
from .records import write_json
from .scoring import lexical_indices


FAMILIES = ("contrastive_negation", "enumerative_preamble")


def _mean(rows, key):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def enrich_spans(spans: list[dict], traces: list[dict], tokenizer) -> list[dict]:
    by_generation = {row["generation_id"]: row for row in traces}
    enriched = []
    for span in spans:
        if (
            span.get("alignment_status") != "ok"
            or span.get("token_start") is None
            or span.get("boundary_token") is None
        ):
            continue
        trace = by_generation.get(span["generation_id"])
        if trace is None:
            continue
        start = int(span["token_start"])
        boundary = int(span["boundary_token"])
        token_ids = trace["token_ids"]
        pre = trace["entropy"][max(0, start - 8) : start]
        lexical_after = [
            boundary + index
            for index in lexical_indices(tokenizer, token_ids[boundary:])
        ]
        if len(pre) != 8 or len(lexical_after) != 12:
            continue
        scaffold_change = float(
            np.mean(pre) - np.mean([trace["entropy"][i] for i in lexical_after])
        )
        control_start = span.get("control_token_start")
        control_end = span.get("control_token_end")
        control_change = float("nan")
        if control_start is not None and control_end is not None:
            control_pre = trace["entropy"][
                max(0, int(control_start) - 8) : int(control_start)
            ]
            control_after_indices = [
                int(control_end) + index
                for index in lexical_indices(
                    tokenizer, token_ids[int(control_end) :]
                )
            ]
            if len(control_pre) == 8 and len(control_after_indices) == 12:
                control_change = float(
                    np.mean(control_pre)
                    - np.mean(
                        [trace["entropy"][index] for index in control_after_indices]
                    )
                )
        row = dict(span)
        row["pre_entropy"] = float(np.mean(pre))
        row["post_entropy"] = float(
            np.mean([trace["entropy"][i] for i in lexical_after])
        )
        row["event_entropy_values"] = [
            *[float(value) for value in pre],
            *[float(trace["entropy"][i]) for i in lexical_after],
        ]
        row["control_entropy_values"] = None
        if (
            control_start is not None
            and control_end is not None
            and math.isfinite(control_change)
        ):
            row["control_entropy_values"] = [
                *[float(value) for value in control_pre],
                *[
                    float(trace["entropy"][index])
                    for index in control_after_indices
                ],
            ]
        row["event_change"] = scaffold_change
        row["control_event_change"] = control_change
        row["event_effect"] = scaffold_change - control_change
        row["span_surprisal"] = float(
            np.mean(trace["surprisal"][start:boundary])
        )
        enriched.append(row)
    return enriched


def _bootstrap_effect(rows: list[dict], key: str) -> dict:
    usable = [
        row
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    if not usable:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "standardized": None,
            "p_value": None,
            "n": 0,
        }
    estimate, low, high = clustered_bootstrap(
        usable, lambda sample: float(_mean(sample, key))
    )
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in usable:
        by_prompt[str(row["prompt_id"])].append(float(row[key]))
    values = np.asarray(
        [np.mean(prompt_values) for prompt_values in by_prompt.values()],
        dtype=float,
    )
    standard_deviation = values.std(ddof=1) if len(values) > 1 else 0.0
    standardized = estimate / standard_deviation if standard_deviation else 0.0
    # Holm reporting uses prompt-level means so repeated responses and multiple
    # events within a prompt do not create pseudo-replication.
    standard_error = standard_deviation / math.sqrt(len(values)) if len(values) > 1 else math.inf
    if math.isfinite(standard_error) and standard_error > 0:
        z = abs(estimate / standard_error)
        p_value = math.erfc(z / math.sqrt(2))
    else:
        p_value = 1.0
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "standardized": float(standardized),
        "p_value": p_value,
        "n": len(usable),
    }


def summarize(
    generations: list[dict], spans: list[dict], traces: list[dict], tokenizer
) -> tuple[dict, list[dict]]:
    inferential = [row for row in generations if not row["greedy"]]
    enriched = enrich_spans(spans, traces, tokenizer)
    by_stage_generation = defaultdict(list)
    for row in inferential:
        by_stage_generation[row["stage"]].append(row)
    stages = [stage for stage in STAGE_ORDER if stage in by_stage_generation]
    stages.extend(
        sorted(stage for stage in by_stage_generation if stage not in STAGE_ORDER)
    )
    stage_metrics: dict[str, dict] = {}
    for stage in stages:
        stage_rows = by_stage_generation[stage]
        token_total = sum(len(row["generated_token_ids"]) for row in stage_rows)
        family_metrics = {}
        for family in FAMILIES:
            hits = [
                row
                for row in enriched
                if row["stage"] == stage
                and row["family"] == family
                and not row["greedy"]
            ]
            hit_responses = {row["generation_id"] for row in hits}
            family_metrics[family] = {
                "hits": len(hits),
                "incidence_per_1000_tokens": (
                    1000 * len(hits) / token_total if token_total else None
                ),
                "response_rate": (
                    len(hit_responses) / len(stage_rows) if stage_rows else None
                ),
                "span_surprisal": _mean(hits, "span_surprisal"),
            }
        stage_metrics[stage] = family_metrics

    primary = {}
    entropy_p = {}
    ablation_p = {}
    for family in FAMILIES:
        rows = [
            row
            for row in enriched
            if row["family"] == family and not row["greedy"]
        ]
        entropy = _bootstrap_effect(rows, "event_effect")
        ablation = _bootstrap_effect(rows, "ablation_gain")
        primary[family] = {"entropy_effect": entropy, "ablation_gain": ablation}
        entropy_p[family] = (
            entropy["p_value"] if entropy["p_value"] is not None else 1.0
        )
        ablation_p[family] = (
            ablation["p_value"] if ablation["p_value"] is not None else 1.0
        )
    entropy_adjusted = holm_adjust(entropy_p)
    ablation_adjusted = holm_adjust(ablation_p)
    for family in FAMILIES:
        primary[family]["entropy_effect"]["holm_p"] = entropy_adjusted[family]
        primary[family]["ablation_gain"]["holm_p"] = ablation_adjusted[family]
        entropy = primary[family]["entropy_effect"]
        ablation = primary[family]["ablation_gain"]
        primary[family]["supports_thesis"] = bool(
            entropy["ci_low"] is not None
            and entropy["ci_low"] > 0
            and ablation["ci_low"] is not None
            and ablation["ci_low"] > 0
        )

    dissociation = {}
    for family in FAMILIES:
        values = {
            stage: stage_metrics[stage][family]["incidence_per_1000_tokens"]
            for stage in ("sft", "dpo", "rlvr")
            if stage in stage_metrics
        }
        if len(values) == 3 and all(value is not None for value in values.values()):
            sft_dpo, dpo_rlvr = stage_deltas(values)
            dissociation[family] = {
                "sft_to_dpo": sft_dpo,
                "dpo_to_rlvr": dpo_rlvr,
            }
    truncations = sum(row["finish_reason"] == "length" for row in generations)
    required_generation_fields = {
        "checkpoint_revision",
        "seed",
        "temperature",
        "top_p",
        "repetition_penalty",
        "max_tokens",
        "finish_reason",
    }
    metadata_failures = sum(
        any(field not in row or row[field] is None for field in required_generation_fields)
        for row in generations
    )
    trace_failures = 0
    for trace in traces:
        arrays = (trace["token_ids"], trace["entropy"], trace["surprisal"])
        if len({len(values) for values in arrays}) != 1 or any(
            not math.isfinite(float(value))
            for values in arrays[1:]
            for value in values
        ):
            trace_failures += 1
    ablation_statuses: dict[str, int] = defaultdict(int)
    for span in spans:
        ablation_statuses[str(span.get("ablation_status", "not_scored"))] += 1
    summary = {
        "stage_metrics": stage_metrics,
        "primary": primary,
        "stage_dissociation": dissociation,
        "quality": {
            "generation_count": len(generations),
            "truncation_count": truncations,
            "truncation_rate": truncations / len(generations) if generations else None,
            "detector_failure_count": sum(
                row.get("alignment_status") != "ok" for row in spans
            ),
            "detector_failure_rate": (
                sum(row.get("alignment_status") != "ok" for row in spans)
                / len(spans)
                if spans
                else 0.0
            ),
            "stage_pairing": validate_stage_pairing(generations),
            "generation_metadata_failure_count": metadata_failures,
            "trace_count": len(traces),
            "trace_integrity_failure_count": trace_failures,
            "ablation_status_counts": dict(sorted(ablation_statuses.items())),
        },
        "decision_rule": (
            "Both prompt-clustered 95% CI lower bounds must exceed zero; "
            "|standardized effect| < 0.1 is negligible and >= 0.2 meaningful."
        ),
    }
    return summary, enriched


def evaluate_audit(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    truthy = {"1", "true", "yes", "y"}
    falsy = {"0", "false", "no", "n"}
    parsed = []
    for row in rows:
        label = row["human_label"].strip().lower()
        if label in truthy | falsy:
            parsed.append((row, label in truthy))
    if len(parsed) != len(rows):
        return {
            "status": "pending_human_labels",
            "labeled": len(parsed),
            "required": len(rows),
        }
    families = {}
    passed = True
    for family in FAMILIES:
        family_rows = [item for item in parsed if item[0]["family"] == family]
        detected = [
            truth
            for row, truth in family_rows
            if row["detector_hit"].lower() == "true"
        ]
        missed = [
            truth
            for row, truth in family_rows
            if row["detector_hit"].lower() == "false"
        ]
        precision = sum(detected) / len(detected) if detected else None
        false_negative_rate = sum(missed) / len(missed) if missed else None
        family_passed = (
            len(detected) >= 50
            and len(missed) >= 50
            and precision is not None
            and precision >= 0.9
        )
        passed &= family_passed
        families[family] = {
            "precision": precision,
            "audited_hits": len(detected),
            "audited_non_hits": len(missed),
            "non_hit_positive_rate": false_negative_rate,
            "passed": family_passed,
        }
    return {"status": "complete", "passed": passed, "families": families}


def make_figures(output: Path, summary: dict, enriched: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stages = list(summary["stage_metrics"])
    x = np.arange(len(stages))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for offset, family in zip((-width / 2, width / 2), FAMILIES):
        incidence = [
            summary["stage_metrics"][stage][family]["incidence_per_1000_tokens"]
            or 0
            for stage in stages
        ]
        surprisal = [
            summary["stage_metrics"][stage][family]["span_surprisal"]
            for stage in stages
        ]
        axes[0].bar(x + offset, incidence, width, label=family)
        axes[1].plot(x, surprisal, marker="o", label=family)
    for axis in axes:
        axis.set_xticks(x, [stage.upper() for stage in stages])
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Events per 1,000 tokens")
    axes[1].set_ylabel("Emitted-token surprisal (nats)")
    fig.tight_layout()
    fig.savefig(figures / "stage_incidence_surprisal.png", dpi=180)
    plt.close(fig)

    offsets = [*range(-8, 0), *range(1, 13)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, family in zip(axes, FAMILIES):
        family_rows = [
            row
            for row in enriched
            if row["family"] == family and row.get("control_entropy_values")
        ]
        if family_rows:
            event = np.asarray(
                [row["event_entropy_values"] for row in family_rows], dtype=float
            )
            control = np.asarray(
                [row["control_entropy_values"] for row in family_rows], dtype=float
            )
            axis.plot(offsets, event.mean(axis=0), label="scaffold")
            axis.plot(offsets, control.mean(axis=0), linestyle="--", label="control")
        else:
            axis.text(
                0.5,
                0.5,
                "No matched events",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlim(-8, 12)
        axis.set_title(family.replace("_", " "))
        axis.set_xlabel("Token offset (post-boundary tokens are lexical)")
        if family_rows:
            axis.legend(fontsize=8)
    axes[0].set_ylabel("Predictive entropy (nats)")
    fig.tight_layout()
    fig.savefig(figures / "event_aligned_entropy.png", dpi=180)
    plt.close(fig)

    for filename, key, ylabel in (
        ("scaffold_ablation_gain.png", "ablation_gain", "Ablation gain vs control (nats)"),
    ):
        data = [
            [
                row[key]
                for row in enriched
                if row["family"] == family and row.get(key) is not None
                and math.isfinite(float(row[key]))
            ]
            for family in FAMILIES
        ]
        fig, axis = plt.subplots(figsize=(6, 4))
        if any(data):
            axis.boxplot(data, tick_labels=["contrast", "enumerative"], showmeans=True)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)


def write_audit_sample(output: Path, generations: list[dict], spans: list[dict]) -> None:
    by_generation = {row["generation_id"]: row for row in generations}
    rows = []
    rng = np.random.default_rng(202707)
    for family in FAMILIES:
        hits = [row for row in spans if row["family"] == family]
        chosen_hits = (
            list(rng.choice(hits, min(50, len(hits)), replace=False)) if hits else []
        )
        hit_ids = {row["generation_id"] for row in hits}
        misses = [row for row in generations if row["generation_id"] not in hit_ids]
        chosen_misses = (
            list(rng.choice(misses, min(50, len(misses)), replace=False))
            if misses
            else []
        )
        for is_hit, samples in ((True, chosen_hits), (False, chosen_misses)):
            for sample in samples:
                generation = by_generation.get(sample["generation_id"], sample)
                rows.append(
                    {
                        "audit_id": f"audit-{len(rows):04d}",
                        "family": family,
                        "detector_hit": is_hit,
                        "text": sample["text"] if is_hit else generation["text"],
                        "human_label": "",
                    }
                )
    rng.shuffle(rows)
    with (output / "detector_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [
            "audit_id", "family", "detector_hit", "text", "human_label"
        ])
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "detector_audit_status.json",
        {
            "status": "pending_human_labels",
            "required_precision": 0.9,
            "requested_hits_and_non_hits_per_family": 50,
            "sample_count": len(rows),
        },
    )
