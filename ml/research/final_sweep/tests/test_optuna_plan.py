from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "optuna_sweep.py"
spec = importlib.util.spec_from_file_location("optuna_sweep", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys

sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_config() -> module.SweepConfig:
    return module.SweepConfig(
        research_python=Path("/research/python"),
        work_root=Path("/work"),
        manifest=Path("/manifest.jsonl"),
        audio_root=Path("/audio"),
        exclude=Path("/exclude.jsonl"),
        eval_slice=Path("/eval.jsonl"),
        public_rows=10_000,
        public_revision="969944574ea3",
        public_steps=10_000,
        aqua_steps=1_000,
        eval_limit=500,
        prompt_ids=("xml_dictation_v1",),
        timeout=3600,
    )


def params() -> dict[str, object]:
    return {
        "prompt_id": "xml_dictation_v1",
        "lr": 0.0001,
        "adapter_lr": 0.0001,
        "rank": 16,
        "lora_dropout": 0.05,
        "context_length": 768,
        "warmup": 100,
    }


def test_prompt_is_baked_into_pack_and_evaluation() -> None:
    plan = module.build_trial_plan(3, params(), make_config())
    assert "--prompt-id" in plan["commands"]["build_public"]
    assert "xml_dictation_v1" in plan["commands"]["build_public"]
    assert "--prompt-id" in plan["commands"]["build_aqua"]
    assert "--prompt-id" in plan["commands"]["evaluate"]
    evaluate = plan["commands"]["evaluate"]
    assert evaluate[evaluate.index("--prompt-id") + 1] == "xml_dictation_v1"


def test_checkpoint_name_uses_python_five_digit_format() -> None:
    plan = module.build_trial_plan(1, params(), make_config())
    assert plan["public_checkpoint"].name == "ckpt_step10000.pt"
    assert plan["prompt_id"] == "xml_dictation_v1"


def test_prompt_hash_is_stable() -> None:
    assert len(module.prompt_sha256("xml_dictation_v1")) == 64
