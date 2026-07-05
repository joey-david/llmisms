from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .generation import chat_prompt


class SequenceScorer(Protocol):
    tokenizer: object

    def prompt_ids(
        self, prompt: str, prompt_format: str = "chat"
    ) -> list[int]: ...

    def trace(self, input_ids: list[int], response_start: int) -> tuple[list[float], list[float]]: ...

    def continuation_surprisals(
        self, prefix_ids: list[int], continuation_ids: list[int]
    ) -> list[float]: ...


def _validate_trace(
    token_ids: list[int], entropy: list[float], surprisal: list[float]
) -> None:
    if not (len(token_ids) == len(entropy) == len(surprisal)):
        raise AssertionError("Token, entropy, and surprisal lengths differ")
    if not all(math.isfinite(value) for value in entropy + surprisal):
        raise AssertionError("Trace contains NaN/Inf")


class MLXScorer:
    def __init__(self, model_path: Path):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, self.tokenizer = load(str(model_path))

    def prompt_ids(
        self, prompt: str, prompt_format: str = "chat"
    ) -> list[int]:
        return list(
            chat_prompt(
                self.tokenizer,
                prompt,
                tokenize=True,
                prompt_format=prompt_format,
            )
        )

    def _arrays(
        self, input_ids: list[int], selected_ids: list[int], first_logit: int
    ) -> tuple[list[float], list[float]]:
        mx = self.mx
        output = self.model(mx.array([input_ids]))
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[0, first_logit : first_logit + len(selected_ids)].astype(
            mx.float32
        )
        log_z = mx.logsumexp(logits, axis=-1)
        probabilities = mx.exp(logits - log_z[:, None])
        entropy = log_z - mx.sum(probabilities * logits, axis=-1)
        selected = mx.array(selected_ids)[:, None]
        chosen = mx.take_along_axis(logits, selected, axis=-1).squeeze(-1)
        surprisal = log_z - chosen
        mx.eval(entropy, surprisal)
        return (
            [float(value) for value in np.asarray(entropy)],
            [float(value) for value in np.asarray(surprisal)],
        )

    def trace(
        self, input_ids: list[int], response_start: int
    ) -> tuple[list[float], list[float]]:
        selected = input_ids[response_start:]
        return self._arrays(input_ids, selected, response_start - 1)

    def continuation_surprisals(
        self, prefix_ids: list[int], continuation_ids: list[int]
    ) -> list[float]:
        _, surprisal = self._arrays(
            prefix_ids + continuation_ids,
            continuation_ids,
            len(prefix_ids) - 1,
        )
        return surprisal


class TransformersScorer:
    def __init__(self, checkpoint: str, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        load_revision = None if Path(checkpoint).exists() else revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=load_revision,
            fix_mistral_regex=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            revision=load_revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def prompt_ids(
        self, prompt: str, prompt_format: str = "chat"
    ) -> list[int]:
        return list(
            chat_prompt(
                self.tokenizer,
                prompt,
                tokenize=True,
                prompt_format=prompt_format,
            )
        )

    def _arrays(
        self, input_ids: list[int], selected_ids: list[int], first_logit: int
    ) -> tuple[list[float], list[float]]:
        torch = self.torch
        device = next(self.model.parameters()).device
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = self.model(tokens, use_cache=False)
            logits = output.logits[
                0, first_logit : first_logit + len(selected_ids)
            ].float()
            log_z = torch.logsumexp(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            entropy = log_z - (probabilities * logits).sum(dim=-1)
            chosen = logits.gather(
                -1,
                torch.tensor(selected_ids, device=logits.device)[:, None],
            ).squeeze(-1)
            surprisal = log_z - chosen
        return entropy.cpu().tolist(), surprisal.cpu().tolist()

    def trace(
        self, input_ids: list[int], response_start: int
    ) -> tuple[list[float], list[float]]:
        selected = input_ids[response_start:]
        return self._arrays(input_ids, selected, response_start - 1)

    def continuation_surprisals(
        self, prefix_ids: list[int], continuation_ids: list[int]
    ) -> list[float]:
        _, surprisal = self._arrays(
            prefix_ids + continuation_ids,
            continuation_ids,
            len(prefix_ids) - 1,
        )
        return surprisal


def lexical_indices(tokenizer, token_ids: list[int], limit: int = 12) -> list[int]:
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    prefix = "".join(pieces)
    marker = re.match(
        r"^\s*(?:(?:\d+|one|first(?:ly)?)\s*[\).,:-]?\s*)+",
        prefix,
        flags=re.IGNORECASE,
    )
    marker_end = marker.end() if marker else 0
    indices: list[int] = []
    char_end = 0
    for index, piece in enumerate(pieces):
        char_end += len(piece)
        if char_end <= marker_end:
            continue
        if any(character.isalnum() for character in piece):
            indices.append(index)
            if len(indices) == limit:
                break
    return indices


def select_control(
    token_count: int,
    span_start: int,
    span_end: int,
    occupied: list[tuple[int, int]],
    entropy: list[float],
    punctuation_starts: set[int] | None = None,
) -> tuple[int, int] | None:
    length = span_end - span_start
    if length <= 0:
        return None
    target_position = span_start / max(1, token_count)
    target_pre = np.mean(entropy[max(0, span_start - 8) : span_start] or [np.nan])
    candidates = []
    for start in range(8, token_count - length - 12):
        if punctuation_starts is not None and start not in punctuation_starts:
            continue
        end = start + length
        if any(start < right and end > left for left, right in occupied):
            continue
        pre = np.mean(entropy[max(0, start - 8) : start] or [np.nan])
        position_cost = abs(start / token_count - target_position)
        entropy_cost = abs(pre - target_pre) if np.isfinite(target_pre) else 0.0
        candidates.append((position_cost + 0.1 * entropy_cost, start, end))
    return min(candidates)[1:] if candidates else None


def score_generation(scorer: SequenceScorer, generation: dict) -> dict:
    prompt_ids = scorer.prompt_ids(
        generation["prompt"], generation.get("prompt_format", "chat")
    )
    generated_ids = [int(value) for value in generation["generated_token_ids"]]
    if generated_ids:
        entropy, surprisal = scorer.trace(
            prompt_ids + generated_ids, len(prompt_ids)
        )
        trace_status = "ok"
    else:
        entropy, surprisal = [], []
        trace_status = "empty_generation"
    _validate_trace(generated_ids, entropy, surprisal)
    return {
        "generation_id": generation["generation_id"],
        "prompt_id": generation["prompt_id"],
        "stage": generation["stage"],
        "token_ids": generated_ids,
        "entropy": entropy,
        "surprisal": surprisal,
        "token_count": len(generated_ids),
        "trace_status": trace_status,
    }


def score_ablation(
    scorer: SequenceScorer,
    generation: dict,
    trace: dict,
    span: dict,
    occupied: list[tuple[int, int]],
) -> dict:
    prompt_ids = scorer.prompt_ids(
        generation["prompt"], generation.get("prompt_format", "chat")
    )
    generated = [int(value) for value in generation["generated_token_ids"]]
    start, boundary = int(span["token_start"]), int(span["boundary_token"])
    continuation = generated[boundary:]
    lexical = lexical_indices(scorer.tokenizer, continuation)
    if len(lexical) < 12:
        return {"ablation_status": "insufficient_lexical_continuation"}
    continuation = continuation[: lexical[-1] + 1]
    with_values = scorer.continuation_surprisals(
        prompt_ids + generated[:boundary], continuation
    )
    deleted_values = scorer.continuation_surprisals(
        prompt_ids + generated[:start], continuation
    )
    scaffold_gain = float(
        np.mean([deleted_values[i] - with_values[i] for i in lexical])
    )

    control = select_control(
        len(generated),
        start,
        boundary,
        occupied,
        trace["entropy"],
        {
            index + 1
            for index, token_id in enumerate(generated[:-1])
            if any(
                mark
                in scorer.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for mark in ".!?;:\n"
            )
        },
    )
    if control is None:
        return {
            "ablation_status": "no_control",
            "scaffold_gain": scaffold_gain,
        }
    control_start, control_end = control
    control_continuation = generated[control_end:]
    control_lexical = lexical_indices(scorer.tokenizer, control_continuation)
    if len(control_lexical) < 12:
        return {
            "ablation_status": "insufficient_control_continuation",
            "scaffold_gain": scaffold_gain,
        }
    control_continuation = control_continuation[: control_lexical[-1] + 1]
    control_with = scorer.continuation_surprisals(
        prompt_ids + generated[:control_end], control_continuation
    )
    control_deleted = scorer.continuation_surprisals(
        prompt_ids + generated[:control_start], control_continuation
    )
    control_gain = float(
        np.mean(
            [
                control_deleted[i] - control_with[i]
                for i in control_lexical
            ]
        )
    )
    return {
        "ablation_status": "ok",
        "scaffold_gain": scaffold_gain,
        "control_gain": control_gain,
        "ablation_gain": scaffold_gain - control_gain,
        "control_token_start": control_start,
        "control_token_end": control_end,
    }
