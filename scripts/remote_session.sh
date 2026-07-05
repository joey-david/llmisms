#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "full" ) ]]; then
  echo "remote_session accepts only smoke or full" >&2
  exit 64
fi
mode=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output="$root/outputs/$mode"
mkdir -p "$output"
cd "$root"

exec 9>"$root/.run.lock"
if ! flock -n 9; then
  echo "Another llmisms remote session is active; refusing to overlap" >&2
  exit 1
fi
run_without_lock() {
  "$@" 9>&-
}

# All bulky state is project-owned so it can be removed without touching
# shared account caches after the final result pull.
printf '%s\n' "llmisms-owned-storage-v1" > "$root/.llmisms-owned-storage"
export XDG_CACHE_HOME="$root/.cache"
export PIP_CACHE_DIR="$root/.cache/pip"
export HF_HOME="$root/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$root/.cache/torch"
export VLLM_CACHE_ROOT="$root/.cache/vllm"
export TRITON_CACHE_DIR="$root/.cache/triton"
export TMPDIR="$root/.cache/tmp"
export UV_CACHE_DIR="$root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/.cache/uv/python"
mkdir -p \
  "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR"
cat > "$output/storage-scope.json" <<EOF
{
  "owner_marker": "$root/.llmisms-owned-storage",
  "removable_paths": [
    "$root/.venv",
    "$root/.venv-cu124",
    "$root/.cache",
    "$root/models",
    "$root/outputs"
  ],
  "shared_caches_used": false
}
EOF

# Record GPU occupancy once for diagnostics, but do not block shared use.
nvidia-smi -q -x > "$output/nvidia-smi.xml"
python3 - "$output/nvidia-smi.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
gpus = root.findall("gpu")
if len(gpus) != 2:
    raise SystemExit(f"Expected exactly two GPUs; found {len(gpus)}")
processes = [item for gpu in gpus for item in gpu.findall("./processes/process_info")]
if processes:
    details = ", ".join(
        f"pid={item.findtext('pid')} name={item.findtext('process_name')}"
        for item in processes
    )
    print(f"WARNING: GPU already has active processes ({details}); continuing")
PY

if [[ "$mode" == "full" ]]; then
  free_kb=$(df -Pk "$root" | awk 'NR==2 {print $4}')
  # df -P reports 1,024-byte blocks; the threshold is 320 decimal GB.
  if (( free_kb < 312500000 )); then
    echo "Full run requires at least 320 GB free storage" >&2
    exit 1
  fi
fi

if [[ ! -x "$root/.venv-cu124/bin/python" ]]; then
  uv venv --python 3.12 --seed "$root/.venv-cu124"
  "$root/.venv-cu124/bin/python" -m pip install --upgrade pip
fi
"$root/.venv-cu124/bin/python" -m pip install -r requirements-upnquick.txt
"$root/.venv-cu124/bin/python" - <<'PY'
import importlib.metadata
import torch
from pathlib import Path

expected = {}
for line in Path("requirements-upnquick.txt").read_text().splitlines():
    if "==" in line:
        name, version = line.split("==", 1)
        expected[name] = version
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Pinned-version mismatch: expected {expected}, got {actual}")
if not torch.version.cuda or not torch.version.cuda.startswith("12.4"):
    raise SystemExit(f"Expected a CUDA 12.4 PyTorch build, got {torch.version.cuda}")
torch.cuda.init()
print(actual, "torch_cuda=", torch.version.cuda)
PY

python_bin="$root/.venv-cu124/bin/python"
sft_model="$root/models/Olmo-3.1-32B-Instruct-SFT-bf16"
if [[ "$mode" == "smoke" ]]; then
  run_without_lock "$python_bin" -m llmisms build-corpus --output "$output" --limit 1
  run_without_lock "$python_bin" -m llmisms generate --output "$output" --backend transformers \
    --stage sft --model "$sft_model" --seeds 1729 --max-tokens 64 --no-greedy
  run_without_lock "$python_bin" -m llmisms tag --output "$output" --backend transformers --stage sft
  run_without_lock "$python_bin" -m llmisms score --output "$output" --backend transformers --stage sft --model "$sft_model"
  "$python_bin" -m pip freeze > "$output/requirements-upnquick.lock.txt"
else
  run_without_lock "$python_bin" -m llmisms build-corpus --output "$output"
  for stage in sft base dpo rlvr; do
    model_args=()
    case "$stage" in
      base) local_model="$root/models/Olmo-3-1125-32B-bf16" ;;
      sft) local_model="$root/models/Olmo-3.1-32B-Instruct-SFT-bf16" ;;
      dpo) local_model="$root/models/Olmo-3.1-32B-Instruct-DPO-bf16" ;;
      rlvr) local_model="$root/models/Olmo-3.1-32B-Instruct-bf16" ;;
    esac
    if [[ -d "$local_model" ]]; then
      model_args=(--model "$local_model")
    fi
    generation_args=()
    if [[ "$stage" == "base" ]]; then
      base_max_tokens=$("$python_bin" - "$output/generations.parquet" <<'PY'
import statistics
import sys
import pyarrow.parquet as pq

rows = pq.read_table(
    sys.argv[1],
    columns=["stage", "greedy", "generated_token_ids"],
).to_pylist()
lengths = [
    len(row["generated_token_ids"])
    for row in rows
    if row["stage"] == "sft" and not row["greedy"]
]
if not lengths:
    raise SystemExit("Cannot set the base cap without SFT generations")
print(round(statistics.median(lengths)))
PY
)
      generation_args=(--max-tokens "$base_max_tokens")
      echo "Base generation cap: $base_max_tokens tokens (SFT median)"
    fi
    run_without_lock "$python_bin" -m llmisms generate --output "$output" --backend transformers \
      --stage "$stage" "${model_args[@]}" "${generation_args[@]}"
  done
  run_without_lock "$python_bin" -m llmisms tag --output "$output" --backend transformers --stage sft
  for stage in base sft dpo rlvr; do
    model_args=()
    case "$stage" in
      base) local_model="$root/models/Olmo-3-1125-32B-bf16" ;;
      sft) local_model="$root/models/Olmo-3.1-32B-Instruct-SFT-bf16" ;;
      dpo) local_model="$root/models/Olmo-3.1-32B-Instruct-DPO-bf16" ;;
      rlvr) local_model="$root/models/Olmo-3.1-32B-Instruct-bf16" ;;
    esac
    if [[ -d "$local_model" ]]; then
      model_args=(--model "$local_model")
    fi
    run_without_lock "$python_bin" -m llmisms score --output "$output" --backend transformers \
      --stage "$stage" "${model_args[@]}"
  done
  run_without_lock "$python_bin" -m llmisms analyze --output "$output" \
    --backend transformers --stage sft
fi
