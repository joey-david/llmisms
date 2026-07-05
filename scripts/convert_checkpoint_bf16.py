#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    destination = args.destination.resolve()
    temporary = destination.with_name(f"{destination.name}.incomplete")
    if destination.exists():
        print(f"BF16 checkpoint already exists: {destination}")
        return
    if temporary.exists():
        raise SystemExit(f"Refusing to overwrite partial conversion: {temporary}")

    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.source,
        fix_mistral_regex=True,
    )
    temporary.mkdir(parents=True)
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(temporary)
    del model

    config_path = temporary / "config.json"
    config = json.loads(config_path.read_text())
    rope_scaling = config.get("rope_scaling", {})
    for key in ("beta_fast", "beta_slow"):
        if key in rope_scaling:
            rope_scaling[key] = float(rope_scaling[key])
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    dtypes: set[str] = set()
    total_bytes = 0
    shards = sorted(temporary.glob("*.safetensors"))
    for shard in shards:
        total_bytes += shard.stat().st_size
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtypes.add(str(handle.get_slice(key).get_dtype()))
    if dtypes != {"BF16"}:
        shutil.rmtree(temporary)
        raise SystemExit(f"Conversion verification failed; dtypes={sorted(dtypes)}")

    manifest = {
        "source": str(args.source),
        "source_revision": args.source_revision,
        "dtype": "bfloat16",
        "shard_count": len(shards),
        "weight_bytes": total_bytes,
    }
    (temporary / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(destination)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
