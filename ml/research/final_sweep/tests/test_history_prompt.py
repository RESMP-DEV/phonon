from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "history_prompt.py"
spec = importlib.util.spec_from_file_location("history_prompt", MODULE_PATH)
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_row(audio: str, raw: str, corrected: str) -> dict[str, str]:
    return {"audio": audio, "raw": raw, "corrected": corrected}


def test_history_retrieval_excludes_target_audio() -> None:
    target = make_row("target", "CUDA kernel launch", "CUDA kernel launch")
    pool = [
        target,
        make_row("same", "CUDA kernel launch", "CUDA kernel launch"),
        make_row("other", "unrelated cooking note", "unrelated cooking note"),
    ]
    contract = module.history_contract(target, pool, count=1)
    assert contract["history_audio"] == ["same"]
    assert "target" not in contract["history_audio"]


def test_history_prompt_is_deterministic_and_contains_no_labels() -> None:
    contract = module.history_contract(
        make_row("target", "deploy the service", "deploy the service"),
        [
            make_row(
                "history",
                "deploy the Kubernetes service",
                "deploy the Kubernetes service",
            )
        ],
        count=1,
    )
    first = module.render_history_prompt(contract)
    second = module.render_history_prompt(contract)
    assert first == second
    assert "<history>" in first
    assert "target" not in first
