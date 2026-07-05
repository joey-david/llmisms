from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


FILES = {
    "prompts": "prompts.parquet",
    "generations": "generations.parquet",
    "traces": "token_traces.parquet",
    "spans": "detected_spans.parquet",
}


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is required; install requirements.txt"
        ) from exc
    return pa, pq


def write_records(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    pa, pq = _arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty record set: {path}")
    keys = sorted({key for row in materialized for key in row})
    normalized = [{key: row.get(key) for key in keys} for row in materialized]
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(normalized), temporary)
    temporary.replace(path)


def read_records(path: Path) -> list[dict[str, Any]]:
    _, pq = _arrow()
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
