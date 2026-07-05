from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

from .analysis import evaluate_audit, make_figures, summarize, write_audit_sample
from .config import CHECKPOINTS, DEFAULT_SEEDS, Sampling, resolve_mlx_snapshot
from .corpus import write_corpus
from .generation import (
    generate_mlx,
    generate_transformers,
    generate_vllm,
    generation_id,
)
from .patterns import run_pattern_study
from .records import FILES, read_records, write_json, write_records
from .scoring import MLXScorer, TransformersScorer, score_ablation, score_generation
from .tagging import align_candidate, detect


def _output(args) -> Path:
    path = Path(args.output).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _merge(path: Path, rows: list[dict], key: str) -> None:
    existing = read_records(path) if path.exists() else []
    incoming_keys = {row[key] for row in rows}
    write_records(path, [row for row in existing if row[key] not in incoming_keys] + rows)


def _remove_records(path: Path, key: str, values: set[str]) -> None:
    if not path.exists():
        return
    remaining = [row for row in read_records(path) if row[key] not in values]
    if remaining:
        write_records(path, remaining)
    else:
        path.unlink()


def command_build_corpus(args) -> None:
    output = _output(args)
    rows = write_corpus(output, args.limit)
    print(f"Wrote {len(rows)} frozen prompts to {output / FILES['prompts']}")


def command_generate(args) -> None:
    output = _output(args)
    prompts = read_records(output / FILES["prompts"])
    sampling = Sampling(max_tokens=args.max_tokens)
    seeds = tuple(args.seeds)
    modes = [(seed, False) for seed in seeds]
    if not args.no_greedy:
        modes.append((0, True))
    expected_ids = {
        generation_id(args.stage, prompt["prompt_id"], seed, greedy)
        for prompt in prompts
        for seed, greedy in modes
    }
    generation_path = output / FILES["generations"]
    reusable_engine = {
        "vllm": "vllm-v0-eager",
        "transformers": "transformers-batched-v2",
    }.get(args.backend)
    if reusable_engine and generation_path.exists():
        existing = [
            row
            for row in read_records(generation_path)
            if row["generation_id"] in expected_ids
            and row.get("generation_engine") == reusable_engine
            and row.get("max_tokens") == sampling.max_tokens
        ]
        if {row["generation_id"] for row in existing} == expected_ids:
            print(
                f"Reusing {len(existing)} completed {args.stage} generations"
            )
            return
    if args.backend == "mlx":
        model = Path(args.model) if args.model else resolve_mlx_snapshot()
        rows = generate_mlx(
            prompts,
            model_path=model,
            stage=args.stage,
            seeds=seeds,
            sampling=sampling,
            include_greedy=not args.no_greedy,
        )
    elif args.backend == "vllm":
        checkpoint = args.model or CHECKPOINTS[args.stage]
        rows = generate_vllm(
            prompts,
            checkpoint=checkpoint,
            stage=args.stage,
            seeds=seeds,
            sampling=sampling,
            include_greedy=not args.no_greedy,
            recovery_path=output / f"vllm-raw-{args.stage}.jsonl.incomplete",
        )
    else:
        checkpoint = args.model or CHECKPOINTS[args.stage]
        transformers_recovery = (
            output / f"transformers-{args.stage}.jsonl.incomplete"
        )
        rows = generate_transformers(
            prompts,
            checkpoint=checkpoint,
            stage=args.stage,
            seeds=seeds,
            sampling=sampling,
            include_greedy=not args.no_greedy,
            batch_size=args.batch_size,
            recovery_path=transformers_recovery,
        )
    existing = read_records(generation_path) if generation_path.exists() else []
    write_records(
        generation_path,
        [row for row in existing if row["stage"] != args.stage] + rows,
    )
    regenerated = {row["generation_id"] for row in rows}
    _remove_records(output / FILES["traces"], "generation_id", regenerated)
    _remove_records(output / FILES["spans"], "generation_id", regenerated)
    scoring_recovery = output / f"scoring-{args.stage}.jsonl.incomplete"
    if scoring_recovery.exists():
        scoring_recovery.unlink()
    recovery_path = output / f"vllm-raw-{args.stage}.jsonl.incomplete"
    if recovery_path.exists():
        recovery_path.unlink()
    transformers_recovery = (
        output / f"transformers-{args.stage}.jsonl.incomplete"
    )
    if transformers_recovery.exists():
        transformers_recovery.unlink()
    finish_reasons = Counter(row["finish_reason"] for row in rows)
    alignment_statuses = Counter(
        row.get("token_alignment_status", "exact") for row in rows
    )
    write_json(
        output / f"generation-{args.stage}.json",
        {
            "backend": args.backend,
            "stage": args.stage,
            "count": len(rows),
            "seeds": list(seeds),
            "sampling": sampling.asdict(),
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "natural_stop_rate": (
                finish_reasons.get("stop", 0) / len(rows) if rows else None
            ),
            "token_alignment_status_counts": dict(
                sorted(alignment_statuses.items())
            ),
        },
    )
    print(f"Wrote {len(rows)} generations")
    if len(rows) == 1:
        print(f"Smoke response: {rows[0]['text']!r}")


def _tokenizer(args, generations: list[dict] | None = None):
    from transformers import AutoTokenizer

    if args.model:
        model = args.model
        revision = None
    elif generations:
        selected = next(
            (row for row in generations if row["stage"] == args.stage),
            generations[0],
        )
        model = selected["checkpoint"]
        revision = selected["checkpoint_revision"]
    elif args.backend == "mlx":
        model = str(resolve_mlx_snapshot())
        revision = None
    else:
        model = CHECKPOINTS[args.stage]
        revision = None
    if Path(model).exists():
        revision = None
    return AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        fix_mistral_regex=True,
    )


def command_tag(args) -> None:
    output = _output(args)
    generations = read_records(output / FILES["generations"])
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    tokenizers = {}
    spans = []
    failures = 0
    for generation in tqdm(
        generations,
        desc="Tagging",
        unit="response",
        disable=False,
        dynamic_ncols=True,
    ):
        key = (
            generation["checkpoint"],
            generation["checkpoint_revision"],
        )
        if key not in tokenizers:
            checkpoint, revision = key
            tokenizers[key] = AutoTokenizer.from_pretrained(
                checkpoint,
                revision=None if Path(checkpoint).exists() else revision,
                fix_mistral_regex=True,
            )
        tokenizer = tokenizers[key]
        encoded_text = list(
            tokenizer.encode(generation["text"], add_special_tokens=False)
        )
        generated_ids = [int(value) for value in generation["generated_token_ids"]]
        exact_sequence = encoded_text == generated_ids
        for ordinal, hit in enumerate(detect(generation["text"])):
            base = {
                "span_id": f"{generation['generation_id']}:{ordinal}",
                "generation_id": generation["generation_id"],
                "prompt_id": generation["prompt_id"],
                "stage": generation["stage"],
                "greedy": generation["greedy"],
                "family": hit.family,
                "char_start": hit.char_start,
                "char_end": hit.char_end,
                "boundary_char": hit.boundary_char,
                "text": hit.text,
            }
            try:
                if not exact_sequence:
                    raise AssertionError(
                        "Decoded response does not exactly re-tokenize to generated IDs"
                    )
                base.update(align_candidate(tokenizer, generation["text"], hit))
                base["alignment_status"] = "ok"
            except (AssertionError, ValueError) as exc:
                failures += 1
                base["token_start"] = None
                base["token_end"] = None
                base["boundary_token"] = None
                base["alignment_status"] = type(exc).__name__
            spans.append(base)
    if spans:
        write_records(output / FILES["spans"], spans)
    else:
        # Parquet requires a schema; a no-hit smoke is an explicit JSON result.
        write_json(output / "detected_spans.empty.json", {"count": 0})
    write_json(
        output / "tagging_summary.json",
        {
            "hits": len(spans),
            "alignment_failures": failures,
            "detector_failure_rate": failures / len(spans) if spans else 0.0,
        },
    )
    write_audit_sample(output, generations, spans)
    print(f"Tagged {len(spans)} spans ({failures} alignment failures)")


def command_score(args) -> None:
    output = _output(args)
    generations = [
        row
        for row in read_records(output / FILES["generations"])
        if row["stage"] == args.stage
    ]
    trace_path = output / FILES["traces"]
    recovery_path = output / f"scoring-{args.stage}.jsonl.incomplete"
    traces = (
        [
            row
            for row in read_records(trace_path)
            if row["stage"] == args.stage
        ]
        if trace_path.exists()
        else []
    )
    if recovery_path.exists():
        with recovery_path.open() as handle:
            for line in handle:
                if line.strip():
                    traces.append(json.loads(line))
    traces = list({row["generation_id"]: row for row in traces}.values())
    completed = {row["generation_id"] for row in traces}
    pending = [
        row for row in generations if row["generation_id"] not in completed
    ]

    if args.backend == "mlx":
        scorer = MLXScorer(Path(args.model) if args.model else resolve_mlx_snapshot())
    else:
        revisions = {row["checkpoint_revision"] for row in generations}
        if len(revisions) != 1:
            raise ValueError(
                f"Expected one exact checkpoint revision for {args.stage}, got {revisions}"
            )
        scorer = TransformersScorer(
            args.model or CHECKPOINTS[args.stage],
            revisions.pop(),
        )
    from tqdm.auto import tqdm

    if pending:
        with recovery_path.open("a") as recovery:
            for generation in tqdm(
                pending,
                desc=f"Teacher forcing ({args.stage})",
                unit="response",
                disable=False,
                dynamic_ncols=True,
            ):
                trace = score_generation(scorer, generation)
                recovery.write(json.dumps(trace, allow_nan=False) + "\n")
                recovery.flush()
                traces.append(trace)
    _merge(trace_path, traces, "generation_id")
    if recovery_path.exists():
        recovery_path.unlink()
    span_path = output / FILES["spans"]
    if span_path.exists():
        spans = read_records(span_path)
        by_generation = {row["generation_id"]: row for row in generations}
        by_trace = {row["generation_id"]: row for row in traces}
        updated = []
        for span in tqdm(
            spans,
            desc=f"Ablations ({args.stage})",
            unit="span",
            disable=False,
            dynamic_ncols=True,
        ):
            generation = by_generation.get(span["generation_id"])
            if (
                generation is None
                or span["alignment_status"] != "ok"
                or span["generation_id"] not in by_trace
            ):
                updated.append(span)
                continue
            occupied = [
                (int(other["token_start"]), int(other["boundary_token"]))
                for other in spans
                if other["generation_id"] == span["generation_id"]
                and other["alignment_status"] == "ok"
            ]
            span.update(
                score_ablation(
                    scorer,
                    generation,
                    by_trace[span["generation_id"]],
                    span,
                    occupied,
                )
            )
            updated.append(span)
        write_records(span_path, updated)
    write_json(
        output / f"scoring-{args.stage}.json",
        {
            "backend": args.backend,
            "stage": args.stage,
            "trace_count": len(traces),
            "exact_full_vocabulary_entropy": True,
            "reduction_dtype": "float32",
        },
    )
    del scorer
    gc.collect()


def command_analyze(args) -> None:
    output = _output(args)
    generations = read_records(output / FILES["generations"])
    traces = read_records(output / FILES["traces"])
    spans = (
        read_records(output / FILES["spans"])
        if (output / FILES["spans"]).exists()
        else []
    )
    tokenizer = _tokenizer(args, generations)
    summary, enriched = summarize(generations, spans, traces, tokenizer)
    summary["quality"]["detector_audit"] = evaluate_audit(
        output / "detector_audit.csv"
    )
    write_json(output / "summary.json", summary)
    make_figures(output, summary, enriched)
    print(json.dumps(summary["primary"], indent=2))


def command_patterns(args) -> None:
    output = _output(args)
    summary, _ = run_pattern_study(
        read_records(output / FILES["generations"]),
        read_records(output / FILES["prompts"]),
        output,
        max_patterns=args.max_patterns,
    )
    print(json.dumps({
        "pattern_counts": summary["pattern_counts"],
        "diversity": summary["diversity"],
    }, indent=2))


def command_smoke(args) -> None:
    output = _output(args)
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.limit = 12
    command_build_corpus(smoke_args)
    smoke_args.backend = "mlx"
    smoke_args.stage = "local"
    smoke_args.seeds = [1729, 2718]
    smoke_args.no_greedy = False
    command_generate(smoke_args)
    command_tag(smoke_args)
    command_score(smoke_args)
    command_analyze(smoke_args)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="llmisms")
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True)
    common.add_argument("--model")
    common.add_argument("--backend", choices=("mlx", "vllm", "transformers"), default="mlx")
    common.add_argument(
        "--stage",
        choices=("base", "sft", "dpo", "rlvr", "local"),
        default="sft",
    )

    build = subparsers.add_parser("build-corpus", parents=[common])
    build.add_argument("--limit", type=int)
    build.set_defaults(function=command_build_corpus)

    generate = subparsers.add_parser("generate", parents=[common])
    generate.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    generate.add_argument("--max-tokens", type=int, default=2048)
    generate.add_argument("--no-greedy", action="store_true")
    generate.add_argument("--batch-size", type=int, default=8)
    generate.set_defaults(function=command_generate)

    tag = subparsers.add_parser("tag", parents=[common])
    tag.set_defaults(function=command_tag)

    score = subparsers.add_parser("score", parents=[common])
    score.set_defaults(function=command_score)

    analyze = subparsers.add_parser("analyze", parents=[common])
    analyze.set_defaults(function=command_analyze)

    patterns = subparsers.add_parser("patterns", parents=[common])
    patterns.add_argument("--max-patterns", type=int, default=40)
    patterns.set_defaults(function=command_patterns)

    smoke = subparsers.add_parser("smoke", parents=[common])
    smoke.add_argument("--max-tokens", type=int, default=2048)
    smoke.add_argument("--batch-size", type=int, default=8)
    smoke.set_defaults(function=command_smoke)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.function(args)
