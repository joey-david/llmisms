from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .records import write_json, write_records
from .tagging import detect


PRIMARY_STAGES = ("sft", "dpo", "rlvr")
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+")
_SEGMENT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING_PREFIX = re.compile(r"^\s*#{1,6}\s+")
_META_OFFER = re.compile(
    r"""(?ix)
    \b(?:if\s+you(?:'d|\s+would)?\s+(?:like|want|wish)
       |let\s+me\s+know
       |feel\s+free
       |i\s+can\s+(?:also\s+)?(?:help|provide|explain|expand|walk))
    """
)

# Content words are deliberately collapsed. The retained vocabulary is the
# grammatical and discourse material needed to recover rhetorical templates.
_FUNCTION_WORDS = frozenset(
    """
    a about above according actually after again against all although also am
    an and another any are around as at be because been before being below
    between both but by can cannot could despite did do does doing down during
    each either enough especially even every few first for from further
    generally had has have having he her here hers herself him himself his how
    however i if in indeed instead into is it its itself just key less may me
    might more most much must my myself neither never no nor not now of off on
    once one only or other ought our ours ourselves out over overall perhaps
    rather same second several she should simply since so some such than that
    the their theirs them themselves then there therefore these they third this
    those though through to too under unless until up very was we were what
    when where whether which while who whom why will with within without would
    yet you your yours yourself yourselves
    """.split()
)
_DISCOURSE_CUES = frozenset(
    """
    actually although answer because but can cannot consider could despite
    even first generally here however if instead key less may might more must
    never no not only overall perhaps point question rather second seem
    several should simply than therefore though while would yet
    """.split()
)
_STRONG_SINGLE_CUES = frozenset(
    "although because but however instead never not rather though while yet".split()
)


def _words(text: str) -> list[str]:
    return [token.lower().replace("’", "'") for token in _WORD.findall(text)]


def _word_prefix(text: str, limit: int) -> str:
    matches = list(itertools.islice(_WORD.finditer(text), limit))
    return text[: matches[-1].end()] if matches else ""


def _skeleton(segment: str) -> list[str]:
    marker = None
    if _LIST_PREFIX.match(segment):
        marker = "<LIST>"
    elif _HEADING_PREFIX.match(segment):
        marker = "<HEADING>"
    atoms: list[str] = [marker] if marker else []
    for token in _words(segment):
        atom = (
            "<NUM>"
            if token.isdigit()
            else token
            if token in _FUNCTION_WORDS
            else "<CONTENT>"
        )
        if atom != "<CONTENT>" or not atoms or atoms[-1] != "<CONTENT>":
            atoms.append(atom)
    return atoms


def extract_patterns(
    text: str, *, max_words: int | None = None
) -> set[tuple[str, tuple[str, ...]]]:
    """Return auditable lexical and abstract discourse patterns in a response."""
    patterns: set[tuple[str, tuple[str, ...]]] = set()
    segments = [segment for segment in _SEGMENT.split(text) if segment.strip()]
    remaining = max_words
    for segment_index, segment in enumerate(segments):
        if remaining is not None and remaining <= 0:
            break
        if (
            segment_index >= len(segments) - 2
            and _META_OFFER.search(segment)
        ):
            continue
        lexical = _words(segment)
        if remaining is not None:
            lexical = lexical[:remaining]
            remaining -= len(lexical)
        skeleton = _skeleton(" ".join(lexical))
        if lexical and _LIST_PREFIX.match(segment):
            skeleton.insert(0, "<LIST>")
        elif lexical and _HEADING_PREFIX.match(segment):
            skeleton.insert(0, "<HEADING>")
        for kind, tokens, low, high in (
            ("lexical", lexical, 3, 6),
            ("skeleton", skeleton, 3, 8),
        ):
            for size in range(low, min(high, len(tokens)) + 1):
                for start in range(len(tokens) - size + 1):
                    phrase = tuple(tokens[start : start + size])
                    if not any(
                        token in _DISCOURSE_CUES
                        or token in {"<LIST>", "<HEADING>"}
                        for token in phrase
                    ):
                        continue
                    if kind == "skeleton" and "<CONTENT>" not in phrase:
                        continue
                    retained = [
                        token
                        for token in phrase
                        if token not in {"<CONTENT>", "<NUM>", "<LIST>", "<HEADING>"}
                    ]
                    if (
                        kind == "skeleton"
                        and len(retained) < 2
                        and not any(token in _STRONG_SINGLE_CUES for token in retained)
                    ):
                        continue
                    patterns.add((kind, phrase))
    return patterns


def _contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    size = len(shorter)
    return any(
        longer[index : index + size] == shorter
        for index in range(len(longer) - size + 1)
    )


def _closed_patterns(
    supports: dict[tuple[str, tuple[str, ...]], set[str]],
) -> list[tuple[str, tuple[str, ...]]]:
    by_signature: dict[tuple[str, frozenset[str]], list[tuple[str, ...]]] = (
        defaultdict(list)
    )
    for (kind, phrase), response_ids in supports.items():
        by_signature[(kind, frozenset(response_ids))].append(phrase)
    retained = []
    for (kind, _), phrases in by_signature.items():
        kept: list[tuple[str, ...]] = []
        for phrase in sorted(phrases, key=lambda value: (-len(value), value)):
            if not any(_contains(longer, phrase) for longer in kept):
                kept.append(phrase)
                retained.append((kind, phrase))
    return retained


def _format_pattern(pattern: tuple[str, tuple[str, ...]]) -> str:
    kind, phrase = pattern
    return f"{kind}:" + " ".join(phrase)


def _paired_interval(
    values: dict[str, dict[str, float]],
    earlier: str,
    later: str,
    *,
    iterations: int = 5000,
    seed: int = 202707,
) -> dict[str, float | int]:
    prompt_ids = sorted(set(values[earlier]) & set(values[later]))
    differences = np.asarray(
        [values[later][key] - values[earlier][key] for key in prompt_ids],
        dtype=float,
    )
    if not len(differences):
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(iterations, len(differences)))
    estimates = differences[sampled].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(differences),
    }


def _sign_flip_p(
    differences: np.ndarray, *, iterations: int = 10000, seed: int
) -> float:
    if not len(differences):
        return 1.0
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = iterations
    while remaining:
        size = min(1000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        exceedances += int(np.sum((signs * differences).mean(axis=1) >= observed))
        remaining -= size
    return (exceedances + 1) / (iterations + 1)


def _bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    adjusted = [1.0] * count
    running = 1.0
    for rank, index in reversed(
        list(enumerate(sorted(range(count), key=p_values.__getitem__), start=1))
    ):
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _remove_scaffolds(text: str) -> str:
    spans = sorted(
        ((hit.char_start, hit.char_end) for hit in detect(text)),
        reverse=True,
    )
    for start, end in spans:
        text = text[:start] + " " + text[end:]
    return text


def _ngram_jaccard(
    first: str, second: str, *, size: int = 4, max_words: int = 256
) -> float | None:
    left = _words(first)
    right = _words(second)
    matched_length = min(len(left), len(right), max_words)
    if matched_length < 32:
        return None
    left = left[:matched_length]
    right = right[:matched_length]
    left_ngrams = {tuple(left[i : i + size]) for i in range(len(left) - size + 1)}
    right_ngrams = {
        tuple(right[i : i + size]) for i in range(len(right) - size + 1)
    }
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else None


def _stage_prompt_values(
    rows: list[dict],
    response_patterns: dict[str, set[tuple[str, tuple[str, ...]]]],
    selected: set[tuple[str, tuple[str, ...]]],
    prompt_caps: dict[str, int],
) -> tuple[dict[str, dict[str, dict[str, float]]], int]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["prompt_id"])].append(row)
    values: dict[str, dict[str, dict[str, float]]] = {
        metric: {stage: {} for stage in PRIMARY_STAGES}
        for metric in (
            "surface_overlap",
            "stripped_surface_overlap",
            "shared_template_probability",
            "template_jaccard",
        )
    }
    excluded_pairs = 0
    for (stage, prompt_id), responses in grouped.items():
        if stage not in PRIMARY_STAGES or len(responses) < 2:
            continue
        metrics = defaultdict(list)
        for first, second in itertools.combinations(responses, 2):
            max_words = prompt_caps[prompt_id]
            surface = _ngram_jaccard(
                first["text"], second["text"], max_words=max_words
            )
            stripped = _ngram_jaccard(
                _remove_scaffolds(first["text"]),
                _remove_scaffolds(second["text"]),
                max_words=max_words,
            )
            if surface is None or stripped is None:
                excluded_pairs += 1
                continue
            left = response_patterns[first["generation_id"]] & selected
            right = response_patterns[second["generation_id"]] & selected
            union = left | right
            metrics["surface_overlap"].append(surface)
            metrics["stripped_surface_overlap"].append(stripped)
            metrics["shared_template_probability"].append(float(bool(left & right)))
            metrics["template_jaccard"].append(
                len(left & right) / len(union) if union else 0.0
            )
        for metric, observations in metrics.items():
            if observations:
                values[metric][stage][prompt_id] = float(np.mean(observations))
    return values, excluded_pairs


def _diversity_summary(
    values: dict[str, dict[str, dict[str, float]]]
) -> dict:
    summary = {}
    for metric, stage_values in values.items():
        stage_means = {
            stage: float(np.mean(list(stage_values[stage].values())))
            for stage in PRIMARY_STAGES
            if stage_values[stage]
        }
        comparisons = {}
        for earlier, later in (("sft", "dpo"), ("dpo", "rlvr"), ("sft", "rlvr")):
            comparisons[f"{earlier}_to_{later}"] = _paired_interval(
                stage_values, earlier, later
            )
        summary[metric] = {
            "stage_means": stage_means,
            "comparisons": comparisons,
        }
    return summary


def run_pattern_study(
    generations: list[dict],
    prompts: list[dict],
    output: Path,
    *,
    max_patterns: int = 40,
) -> tuple[dict, list[dict]]:
    known_prompt_ids = {str(row["prompt_id"]) for row in prompts}
    stochastic = [
        row
        for row in generations
        if not row["greedy"] and row["stage"] in {"base", *PRIMARY_STAGES}
    ]
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    for row in stochastic:
        if row["prompt_id"] not in known_prompt_ids:
            raise ValueError(f"Unknown prompt ID: {row['prompt_id']}")
        if row["stage"] in PRIMARY_STAGES:
            prompt_lengths[row["prompt_id"]].append(len(_words(row["text"])))
    prompt_caps = {
        prompt_id: min(256, min(lengths))
        for prompt_id, lengths in prompt_lengths.items()
        if lengths and min(lengths) >= 32
    }
    excluded_prompt_count = len(prompt_lengths) - len(prompt_caps)
    stochastic = [
        row for row in stochastic if row["prompt_id"] in prompt_caps
    ]
    usable_prompt_ids = set(prompt_caps)
    response_patterns = {
        row["generation_id"]: extract_patterns(
            row["text"], max_words=prompt_caps[row["prompt_id"]]
        )
        for row in stochastic
    }
    primary_rows = [
        row for row in stochastic if row["stage"] in PRIMARY_STAGES
    ]

    supports: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    prompt_supports: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    for row in primary_rows:
        for pattern in response_patterns[row["generation_id"]]:
            supports[pattern].add(row["generation_id"])
            prompt_supports[pattern].add(row["prompt_id"])
    eligible = {
        pattern: response_ids
        for pattern, response_ids in supports.items()
        if len(response_ids) >= 16 and len(prompt_supports[pattern]) >= 10
    }
    closed = _closed_patterns(eligible)

    def stage_rate(pattern, rows):
        by_stage = Counter(row["stage"] for row in rows)
        hits = Counter(
            row["stage"]
            for row in rows
            if pattern in response_patterns[row["generation_id"]]
        )
        return {
            stage: hits[stage] / by_stage[stage] if by_stage[stage] else 0.0
            for stage in ("base", *PRIMARY_STAGES)
        }

    rates_by_pattern = {
        pattern: stage_rate(pattern, stochastic) for pattern in closed
    }
    ranked_candidates = sorted(
        closed,
        key=lambda pattern: (len(supports[pattern]), len(prompt_supports[pattern])),
        reverse=True,
    )
    ranked = []
    for pattern in ranked_candidates:
        if any(
            len(supports[pattern] & supports[prior])
            / len(supports[pattern] | supports[prior])
            >= 0.85
            for prior in ranked
        ):
            continue
        ranked.append(pattern)
        if len(ranked) == max_patterns:
            break

    result_rows = []
    p_values = []
    for ordinal, pattern in enumerate(ranked):
        rates = rates_by_pattern[pattern]
        prompt_values = {stage: {} for stage in PRIMARY_STAGES}
        for stage in PRIMARY_STAGES:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in primary_rows:
                if row["stage"] == stage:
                    grouped[row["prompt_id"]].append(row)
            for prompt_id, rows in grouped.items():
                prompt_values[stage][prompt_id] = float(
                    np.mean(
                        [
                            pattern in response_patterns[row["generation_id"]]
                            for row in rows
                        ]
                    )
                )
        test = _paired_interval(
            prompt_values,
            "sft",
            "rlvr",
            seed=202707 + ordinal,
        )
        prompt_ids = sorted(
            set(prompt_values["sft"]) & set(prompt_values["rlvr"])
        )
        differences = np.asarray(
            [
                prompt_values["rlvr"][key] - prompt_values["sft"][key]
                for key in prompt_ids
            ],
            dtype=float,
        )
        p_value = _sign_flip_p(
            differences,
            seed=int.from_bytes(
                hashlib.sha256(_format_pattern(pattern).encode()).digest()[:4],
                "big",
            ),
        )
        p_values.append(p_value)
        result_rows.append(
            {
                "kind": pattern[0],
                "pattern": " ".join(pattern[1]),
                "pooled_frequency_rank": ordinal + 1,
                "pooled_response_support": len(supports[pattern]),
                "pooled_prompt_support": len(prompt_supports[pattern]),
                "base_rate": rates["base"],
                "sft_rate": rates["sft"],
                "dpo_rate": rates["dpo"],
                "rlvr_rate": rates["rlvr"],
                "sft_to_rlvr": test["estimate"],
                "ci_low": test["ci_low"],
                "ci_high": test["ci_high"],
                "p_value": p_value,
                "q_value": None,
                "confirmed": False,
                "prompt_count": test["n"],
            }
        )
    for row, q_value in zip(result_rows, _bh_adjust(p_values)):
        row["q_value"] = q_value
        row["confirmed"] = bool(row["ci_low"] > 0 and q_value < 0.05)

    selected = set(ranked)
    diversity_values, excluded_pairs = _stage_prompt_values(
        primary_rows, response_patterns, selected, prompt_caps
    )
    diversity = _diversity_summary(diversity_values)
    original = diversity_values["surface_overlap"]
    stripped = diversity_values["stripped_surface_overlap"]
    contribution = {
        stage: {
            prompt_id: original[stage][prompt_id] - stripped[stage][prompt_id]
            for prompt_id in set(original[stage]) & set(stripped[stage])
        }
        for stage in PRIMARY_STAGES
    }
    diversity["scaffold_contribution_to_sft_rlvr_gap"] = _paired_interval(
        contribution, "sft", "rlvr"
    )

    detector_counts = {}
    detector_prompt_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {stage: {} for stage in PRIMARY_STAGES}
    )
    for stage in ("base", *PRIMARY_STAGES):
        rows = [row for row in stochastic if row["stage"] == stage]
        family_responses: dict[str, set[str]] = defaultdict(set)
        family_hits = Counter()
        by_prompt: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_prompt[row["prompt_id"]].append(row)
            text = _word_prefix(row["text"], prompt_caps[row["prompt_id"]])
            for hit in detect(text):
                family_hits[hit.family] += 1
                family_responses[hit.family].add(row["generation_id"])
        if stage in PRIMARY_STAGES:
            families = {
                hit.family
                for row in rows
                for hit in detect(
                    _word_prefix(row["text"], prompt_caps[row["prompt_id"]])
                )
            }
            for family in families:
                for prompt_id, prompt_rows in by_prompt.items():
                    detector_prompt_values[family][stage][prompt_id] = float(
                        np.mean(
                            [
                                any(
                                    hit.family == family
                                    for hit in detect(
                                        _word_prefix(
                                            row["text"], prompt_caps[prompt_id]
                                        )
                                    )
                                )
                                for row in prompt_rows
                            ]
                        )
                    )
        detector_counts[stage] = {
            family: {
                "hits": family_hits[family],
                "response_rate": len(response_ids) / len(rows) if rows else None,
            }
            for family, response_ids in sorted(family_responses.items())
        }
    detector_trends = {}
    for family, stage_values in detector_prompt_values.items():
        for stage in PRIMARY_STAGES:
            for prompt_id in usable_prompt_ids:
                stage_values[stage].setdefault(prompt_id, 0.0)
        detector_trends[family] = {
            "stage_rates": {
                stage: float(np.mean(list(stage_values[stage].values())))
                for stage in PRIMARY_STAGES
            },
            "comparisons": {
                f"{earlier}_to_{later}": _paired_interval(
                    stage_values, earlier, later
                )
                for earlier, later in (
                    ("sft", "dpo"),
                    ("dpo", "rlvr"),
                    ("sft", "rlvr"),
                )
            },
        }

    concentration = {}
    closed_set = set(closed)
    for stage in PRIMARY_STAGES:
        counts = Counter()
        for row in primary_rows:
            if row["stage"] != stage:
                continue
            counts.update(response_patterns[row["generation_id"]] & closed_set)
        frequencies = np.asarray(list(counts.values()), dtype=float)
        probabilities = frequencies / frequencies.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        concentration[stage] = {
            "observed_patterns": len(counts),
            "pattern_occurrences": int(frequencies.sum()),
            "effective_patterns": float(math.exp(entropy)),
            "normalized_entropy": (
                entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
            ),
            "top_10_share": float(
                sum(sorted(counts.values(), reverse=True)[:10])
                / frequencies.sum()
            ),
        }

    frequent_by_stage = {}
    for stage in PRIMARY_STAGES:
        frequent_by_stage[stage] = [
            {
                "kind": pattern[0],
                "pattern": " ".join(pattern[1]),
                "response_rate": rates_by_pattern[pattern][stage],
            }
            for pattern in sorted(
                closed,
                key=lambda value: rates_by_pattern[value][stage],
                reverse=True,
            )[:15]
        ]
    summary = {
        "design": {
            "usable_prompts": len(usable_prompt_ids),
            "stochastic_responses": len(stochastic),
            "excluded_short_prompts": excluded_prompt_count,
            "word_budget": (
                "Per-prompt minimum across post-trained samples, capped at 256"
            ),
            "selection": (
                "Pooled frequency without stage labels; invariant to stage-label "
                "permutation"
            ),
            "minimum_pooled_response_support": 16,
            "minimum_pooled_prompt_support": 10,
            "tested_patterns": len(result_rows),
            "primary_stages": list(PRIMARY_STAGES),
        },
        "pattern_counts": {
            "eligible": len(eligible),
            "closed": len(closed),
            "confirmed": sum(row["confirmed"] for row in result_rows),
        },
        "confirmed_patterns": [
            row for row in result_rows if row["confirmed"]
        ],
        "most_frequent_patterns": frequent_by_stage,
        "detector_counts": detector_counts,
        "detector_trends": detector_trends,
        "template_concentration": concentration,
        "diversity": diversity,
        "excluded_short_response_pairs": excluded_pairs,
        "interpretation_boundary": (
            "Surface and rhetorical concentration are not semantic mode collapse."
        ),
    }
    write_records(output / "recurrent_patterns.parquet", result_rows)
    write_json(output / "pattern_summary.json", summary)
    make_pattern_figures(output, result_rows, diversity)
    return summary, result_rows


def make_pattern_figures(
    output: Path, rows: list[dict], diversity: dict
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    strongest = sorted(rows, key=lambda row: row["sft_to_rlvr"], reverse=True)[:12]
    if strongest:
        labels = [row["pattern"][:45] for row in strongest][::-1]
        y = np.arange(len(labels))
        fig, axis = plt.subplots(figsize=(9, 6))
        for offset, stage in zip((-0.22, 0.0, 0.22), PRIMARY_STAGES):
            axis.barh(
                y + offset,
                [row[f"{stage}_rate"] for row in strongest][::-1],
                0.2,
                label=stage.upper(),
            )
        axis.set_yticks(y, labels, fontsize=8)
        axis.set_xlabel("Length-matched response rate")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "recurrent_template_enrichment.png", dpi=180)
        plt.close(fig)

    metrics = (
        "surface_overlap",
        "stripped_surface_overlap",
        "template_jaccard",
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.8))
    for axis, metric in zip(axes, metrics):
        means = diversity[metric]["stage_means"]
        axis.plot(
            range(len(PRIMARY_STAGES)),
            [means[stage] for stage in PRIMARY_STAGES],
            marker="o",
        )
        axis.set_xticks(range(len(PRIMARY_STAGES)), [s.upper() for s in PRIMARY_STAGES])
        axis.set_title(metric.replace("_", " "), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "matched_diversity.png", dpi=180)
    plt.close(fig)
