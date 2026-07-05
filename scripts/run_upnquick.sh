#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "full" ) ]]; then
  echo "usage: $0 {smoke|full}" >&2
  exit 64
fi
mode=$1
host=upnquick
remote_root=/home/lamsade/jdavid/llmisms
local_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir="$local_root/remote-results/$mode"
log_dir="$local_root/remote-results/logs"
command -v ssh >/dev/null
command -v rsync >/dev/null
for required in pyproject.toml requirements-upnquick.txt scripts/remote_session.sh; do
  if [[ ! -f "$local_root/$required" ]]; then
    echo "missing required input: $local_root/$required" >&2
    exit 66
  fi
done
mkdir -p "$result_dir" "$log_dir"

ssh -G "$host" >/dev/null
rsync -az --delete \
  --exclude .git/ \
  --exclude '.venv*/' \
  --exclude .cache/ \
  --exclude .run.lock \
  --exclude .llmisms-owned-storage \
  --exclude __pycache__/ \
  --exclude .pytest_cache/ \
  --exclude outputs/ \
  --exclude models/ \
  --exclude remote-results/ \
  --exclude '*.log' \
  "$local_root/" "$host:$remote_root/"

interrupted=0
trap 'interrupted=1' INT TERM
set +e
ssh -T "$host" "bash '$remote_root/scripts/remote_session.sh' '$mode'" \
  2>&1 | tee "$log_dir/$mode.log"
remote_status=${PIPESTATUS[0]}
set -e
if (( interrupted )); then
  remote_status=130
fi

# Exactly one result pull, including logs and partial diagnostics after failure.
rsync -az "$host:$remote_root/outputs/$mode/" "$result_dir/"
if [[ "$remote_status" -eq 0 && "$mode" == "smoke" ]]; then
  lockfile="$result_dir/requirements-upnquick.lock.txt"
  if [[ ! -f "$lockfile" ]]; then
    echo "remote smoke succeeded without a dependency lockfile" >&2
    exit 1
  fi
  cp "$lockfile" "$local_root/requirements-upnquick.lock.txt"
fi
exit "$remote_status"
