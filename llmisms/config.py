from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


CHECKPOINTS = {
    "base": "allenai/Olmo-3-1125-32B",
    "sft": "allenai/Olmo-3.1-32B-Instruct-SFT",
    "dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
    "rlvr": "allenai/Olmo-3.1-32B-Instruct",
}
STAGE_ORDER = ("base", "sft", "dpo", "rlvr")
DEFAULT_SEEDS = (1729, 2718, 3141)
LOCAL_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--SmolLM3-3B-bf16"
)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 2048

    def asdict(self) -> dict[str, float | int]:
        return asdict(self)


def resolve_mlx_snapshot(path: Path = LOCAL_MODEL) -> Path:
    snapshots = path / "snapshots"
    candidates = sorted(p for p in snapshots.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No cached MLX snapshot below {snapshots}")
    return candidates[-1]
