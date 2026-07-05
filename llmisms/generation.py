from __future__ import annotations

import time
import uuid
import json
import os
from pathlib import Path
from typing import Iterable

from .config import Sampling
from .math_utils import derived_seed


def chat_prompt(
    tokenizer, text: str, *, tokenize: bool, prompt_format: str = "chat"
):
    if prompt_format == "transcript":
        transcript = f"User: {text}\nAssistant:"
        return (
            tokenizer.encode(transcript, add_special_tokens=True)
            if tokenize
            else transcript
        )
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
    )


def generation_id(stage: str, prompt_id: str, base_seed: int, greedy: bool) -> str:
    key = f"llmisms:{stage}:{prompt_id}:{base_seed}:{int(greedy)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def textual_token_ids(
    tokenizer,
    text: str,
    emitted_ids: list[int],
    terminal_ids: Iterable[int] = (),
) -> tuple[list[int], list[int], str, str]:
    special_ids = set(tokenizer.all_special_ids) | set(terminal_ids)
    token_ids = list(emitted_ids)
    stripped: list[int] = []
    while token_ids and token_ids[-1] in special_ids:
        stripped.insert(0, token_ids.pop())
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = list(tokenizer.encode(decoded, add_special_tokens=False))
    if encoded == token_ids:
        status = "terminal_special_stripped" if stripped else "exact"
    else:
        status = "noncanonical_tokenization"
    return token_ids, stripped, status, decoded


def generate_mlx(
    prompts: Iterable[dict],
    *,
    model_path: Path,
    stage: str,
    seeds: tuple[int, ...],
    sampling: Sampling,
    include_greedy: bool = True,
) -> list[dict]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    from tqdm.auto import tqdm

    prompts = list(prompts)
    model, tokenizer = load(str(model_path))
    rows: list[dict] = []
    modes = [(seed, False) for seed in seeds]
    if include_greedy:
        modes.append((0, True))
    prompt_ids = [
        chat_prompt(tokenizer, prompt["text"], tokenize=True)
        for prompt in prompts
    ]
    for batch_index, (seed, greedy) in enumerate(
        tqdm(
            modes,
            desc="MLX generation",
            unit="batch",
            disable=False,
            dynamic_ncols=True,
        ),
        1,
    ):
        actual_seed = derived_seed(seed, "mlx-batch")
        mx.random.seed(actual_seed)
        sampler = make_sampler(
            temp=0.0 if greedy else sampling.temperature,
            top_p=0.0 if greedy else sampling.top_p,
        )
        response = batch_generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=sampling.max_tokens,
            sampler=sampler,
            completion_batch_size=len(prompts),
            prefill_batch_size=len(prompts),
            verbose=False,
        )
        for prompt, text in zip(prompts, response.texts, strict=True):
            token_ids = list(tokenizer.encode(text, add_special_tokens=False))
            if list(tokenizer.encode(text, add_special_tokens=False)) != token_ids:
                raise AssertionError(
                    "MLX response text does not exactly re-tokenize to emitted IDs"
                )
            finish_reason = (
                "length" if len(token_ids) >= sampling.max_tokens else "stop"
            )
            rows.append(
                {
                    "generation_id": generation_id(
                        stage, prompt["prompt_id"], seed, greedy
                    ),
                    "prompt_id": prompt["prompt_id"],
                    "stratum": prompt["stratum"],
                    "stage": stage,
                    "checkpoint": str(model_path),
                    "checkpoint_revision": model_path.name,
                    "generation_engine": "mlx",
                    "prompt_format": "chat",
                    "seed": actual_seed,
                    "base_seed": seed,
                    "greedy": greedy,
                    "temperature": 0.0 if greedy else sampling.temperature,
                    "top_p": 1.0 if greedy else sampling.top_p,
                    "repetition_penalty": sampling.repetition_penalty,
                    "max_tokens": sampling.max_tokens,
                    "finish_reason": finish_reason,
                    "prompt": prompt["text"],
                    "text": text,
                    "generated_token_ids": list(token_ids),
                    "created_unix": time.time(),
                }
            )
        print(f"generation batch {batch_index}/{len(modes)}", flush=True)
    return rows


def generate_vllm(
    prompts: Iterable[dict],
    *,
    checkpoint: str,
    stage: str,
    seeds: tuple[int, ...],
    sampling: Sampling,
    include_greedy: bool = True,
    recovery_path: Path | None = None,
) -> list[dict]:
    os.environ["VLLM_USE_V1"] = "0"
    from huggingface_hub import model_info
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        manifest = json.loads(
            (checkpoint_path / "conversion_manifest.json").read_text()
        )
        revision = manifest["source_revision"]
        load_revision = None
    else:
        revision = model_info(checkpoint).sha
        load_revision = revision
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=load_revision,
        fix_mistral_regex=True,
    )
    llm = LLM(
        model=checkpoint,
        revision=load_revision,
        tokenizer_revision=load_revision,
        tensor_parallel_size=2,
        dtype="bfloat16",
        model_impl="transformers",
        max_model_len=32768,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    requests = []
    prompt_format = "transcript" if stage == "base" else "chat"
    for prompt in prompts:
        formatted = chat_prompt(
            tokenizer,
            prompt["text"],
            tokenize=False,
            prompt_format=prompt_format,
        )
        modes = [(seed, False) for seed in seeds]
        if include_greedy:
            modes.append((0, True))
        for seed, greedy in modes:
            actual_seed = derived_seed(seed, prompt["prompt_id"])
            params = SamplingParams(
                temperature=0.0 if greedy else sampling.temperature,
                top_p=1.0 if greedy else sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=actual_seed,
                repetition_penalty=sampling.repetition_penalty,
            )
            requests.append((prompt, seed, greedy, actual_seed, formatted, params))
    results = llm.generate(
        [request[4] for request in requests],
        [request[5] for request in requests],
        use_tqdm=True,
    )
    rows: list[dict] = []
    recovery_handle = recovery_path.open("w") if recovery_path else None
    try:
        for request, request_result in zip(requests, results, strict=True):
            prompt, seed, greedy, actual_seed, _, _ = request
            result = request_result.outputs[0]
            if recovery_handle:
                recovery_handle.write(
                    json.dumps(
                        {
                            "prompt_id": prompt["prompt_id"],
                            "base_seed": seed,
                            "greedy": greedy,
                            "text": result.text,
                            "token_ids": list(result.token_ids),
                            "finish_reason": result.finish_reason,
                        }
                    )
                    + "\n"
                )
                recovery_handle.flush()
            (
                token_ids,
                stripped_special_ids,
                alignment_status,
                decoded_text,
            ) = textual_token_ids(tokenizer, result.text, list(result.token_ids))
            rows.append(
                {
                "generation_id": generation_id(
                    stage, prompt["prompt_id"], seed, greedy
                ),
                "prompt_id": prompt["prompt_id"],
                "stratum": prompt["stratum"],
                "stage": stage,
                "checkpoint": checkpoint,
                "checkpoint_revision": revision,
                "generation_engine": "vllm-v0-eager",
                "prompt_format": prompt_format,
                "seed": actual_seed,
                "base_seed": seed,
                "greedy": greedy,
                "temperature": 0.0 if greedy else sampling.temperature,
                "top_p": 1.0 if greedy else sampling.top_p,
                "repetition_penalty": sampling.repetition_penalty,
                "max_tokens": sampling.max_tokens,
                "finish_reason": result.finish_reason,
                "prompt": prompt["text"],
                "text": decoded_text,
                "vllm_text_matches_decoded_tokens": result.text == decoded_text,
                "generated_token_ids": token_ids,
                "stripped_terminal_special_token_ids": stripped_special_ids,
                "token_alignment_status": alignment_status,
                "created_unix": time.time(),
                }
            )
            print(f"generation {len(rows)}/{len(requests)}", flush=True)
    finally:
        if recovery_handle:
            recovery_handle.close()
    return rows


def generate_transformers(
    prompts: Iterable[dict],
    *,
    checkpoint: str,
    stage: str,
    seeds: tuple[int, ...],
    sampling: Sampling,
    include_greedy: bool = True,
    batch_size: int = 8,
    recovery_path: Path | None = None,
) -> list[dict]:
    import torch
    from huggingface_hub import model_info
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = list(prompts)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        manifest = json.loads(
            (checkpoint_path / "conversion_manifest.json").read_text()
        )
        revision = manifest["source_revision"]
        load_revision = None
    else:
        revision = model_info(checkpoint).sha
        load_revision = revision
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=load_revision,
        fix_mistral_regex=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=load_revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    configured_eos = model.generation_config.eos_token_id
    if isinstance(configured_eos, int):
        configured_eos = [configured_eos]
    terminal_ids = {
        *tokenizer.all_special_ids,
        *(configured_eos or []),
        tokenizer.pad_token_id,
    }
    modes = [(seed, False) for seed in seeds]
    prompt_format = "transcript" if stage == "base" else "chat"
    if include_greedy:
        modes.append((0, True))
    total_batches = len(modes) * ((len(prompts) + batch_size - 1) // batch_size)
    progress = tqdm(
        total=total_batches,
        desc=f"Transformers generation ({stage})",
        unit="batch",
        dynamic_ncols=True,
    )
    recovered: dict[str, dict] = {}
    if recovery_path and recovery_path.exists():
        expected_ids = {
            generation_id(stage, prompt["prompt_id"], seed, greedy)
            for prompt in prompts
            for seed, greedy in modes
        }
        with recovery_path.open() as handle:
            recovered = {
                row["generation_id"]: row
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
                if row["generation_id"] in expected_ids
            }
        for row in recovered.values():
            token_ids, stripped, status, text = textual_token_ids(
                tokenizer,
                row["text"],
                row["generated_token_ids"],
                terminal_ids,
            )
            row.update(
                generated_token_ids=token_ids,
                stripped_terminal_special_token_ids=stripped,
                token_alignment_status=status,
                text=text,
                finish_reason="stop" if stripped else row["finish_reason"],
                generation_engine="transformers-batched-v2",
                max_tokens=sampling.max_tokens,
            )
    rows: list[dict] = list(recovered.values())
    recovery = recovery_path.open("a") if recovery_path else None
    try:
        for seed, greedy in modes:
            for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
                batch = [
                    prompt
                    for prompt in prompts[start : start + batch_size]
                    if generation_id(
                        stage, prompt["prompt_id"], seed, greedy
                    )
                    not in recovered
                ]
                if not batch:
                    progress.update()
                    continue
                batch_seed = derived_seed(seed, f"transformers-batch-{batch_index}")
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                formatted = [
                    chat_prompt(
                        tokenizer,
                        prompt["text"],
                        tokenize=False,
                        prompt_format=prompt_format,
                    )
                    for prompt in batch
                ]
                inputs = tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding=True,
                ).to(input_device)
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        do_sample=not greedy,
                        temperature=None if greedy else sampling.temperature,
                        top_p=None if greedy else sampling.top_p,
                        repetition_penalty=sampling.repetition_penalty,
                        max_new_tokens=sampling.max_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = sequences[:, inputs["input_ids"].shape[1] :]
                for prompt, emitted in zip(batch, generated, strict=True):
                    emitted_ids = emitted.tolist()
                    (
                        token_ids,
                        stripped_special_ids,
                        alignment_status,
                        decoded_text,
                    ) = textual_token_ids(
                        tokenizer, "", emitted_ids, terminal_ids
                    )
                    finish_reason = (
                        "length"
                        if len(emitted_ids) >= sampling.max_tokens
                        and not stripped_special_ids
                        else "stop"
                    )
                    row = {
                            "generation_id": generation_id(
                                stage, prompt["prompt_id"], seed, greedy
                            ),
                            "prompt_id": prompt["prompt_id"],
                            "stratum": prompt["stratum"],
                            "stage": stage,
                            "checkpoint": checkpoint,
                            "checkpoint_revision": revision,
                            "generation_engine": "transformers-batched-v2",
                            "prompt_format": prompt_format,
                            "seed": batch_seed,
                            "base_seed": seed,
                            "greedy": greedy,
                            "temperature": (
                                0.0 if greedy else sampling.temperature
                            ),
                            "top_p": 1.0 if greedy else sampling.top_p,
                            "repetition_penalty": sampling.repetition_penalty,
                            "max_tokens": sampling.max_tokens,
                            "finish_reason": finish_reason,
                            "prompt": prompt["text"],
                            "text": decoded_text,
                            "generated_token_ids": token_ids,
                            "stripped_terminal_special_token_ids": (
                                stripped_special_ids
                            ),
                            "token_alignment_status": alignment_status,
                            "created_unix": time.time(),
                    }
                    rows.append(row)
                    if recovery:
                        recovery.write(json.dumps(row, allow_nan=False) + "\n")
                        recovery.flush()
                progress.update()
    finally:
        progress.close()
        if recovery:
            recovery.close()
    return rows
