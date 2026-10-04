from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "optuna_sweep.py"
spec = importlib.util.spec_from_file_location("optuna_sweep", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys

sys.modules[spec.name] = module
spec.loader.exec_module(module)

FORMAT_PATH = HERE.parent / "format_bakeoff.py"
format_spec = importlib.util.spec_from_file_location("format_bakeoff", FORMAT_PATH)
format_module = importlib.util.module_from_spec(format_spec)
assert format_spec is not None and format_spec.loader is not None
sys.modules[format_spec.name] = format_module
format_spec.loader.exec_module(format_module)

SYNC_PATH = HERE.parent / "sync_wandb_offline.py"
sync_spec = importlib.util.spec_from_file_location("sync_wandb_offline", SYNC_PATH)
sync_module = importlib.util.module_from_spec(sync_spec)
assert sync_spec.loader is not None
sys.modules[sync_spec.name] = sync_module
sync_spec.loader.exec_module(sync_module)


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


def test_evaluation_receives_trial_lora_rank() -> None:
    """A rank-32 trial must not be reconstructed as the rank-16 default."""

    plan = module.build_trial_plan(
        3, {**params(), "rank": 32}, make_config()
    )
    evaluate = plan["commands"]["evaluate"]
    assert evaluate[evaluate.index("--lora-rank") + 1] == "32"


def test_prompt_hash_is_stable() -> None:
    assert len(module.prompt_sha256("xml_dictation_v1")) == 64


def test_pack_cache_key_includes_context_length() -> None:
    """Packs are context-length specific, so a shared path would silently
    reuse a pack built for a different context window."""

    short = module.build_trial_plan(0, params(), make_config())
    longer = module.build_trial_plan(
        0, {**params(), "context_length": 1024}, make_config()
    )
    assert short["public_pack"] != longer["public_pack"]
    assert short["aqua_pack"] != longer["aqua_pack"]
    assert "ctx768" in str(short["public_pack"])
    assert "ctx1024" in str(longer["public_pack"])


def test_pack_is_valid_rejects_wrong_context_length(tmp_path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dataset_info.json").write_text("{}")
    (pack / "pack_meta.json").write_text(
        '{"prompt_id": "xml_dictation_v1", "context_length": 512}'
    )
    assert not module.pack_is_valid(pack, "xml_dictation_v1", 1024)


def test_format_bakeoff_fixes_best_hyperparameters() -> None:
    assert format_module.BEST_TRIAL_4["rank"] == 32
    assert format_module.BEST_TRIAL_4["context_length"] == 512
    assert format_module.BEST_TRIAL_4["lr"] == 6.505720091093967e-05
    assert format_module.BEST_TRIAL_4["adapter_lr"] == 4.943429131224935e-05
    assert format_module.BEST_TRIAL_4["warmup"] == 50


def test_wandb_uses_one_run_id_across_training_stages() -> None:
    config = replace(
        make_config(),
        wandb_project="phonon",
        wandb_mode="offline",
    )
    plan = module.build_trial_plan(100, params(), config)
    public = plan["commands"]["train_public"]
    aqua = plan["commands"]["train_aqua"]

    assert plan["wandb_run_id"] == "format-0100-xml_dictation_v1"
    assert public[public.index("--wandb-run-id") + 1] == plan["wandb_run_id"]
    assert aqua[aqua.index("--wandb-run-id") + 1] == plan["wandb_run_id"]
    assert public[public.index("--wandb-stage") + 1] == "public"
    assert aqua[aqua.index("--wandb-stage") + 1] == "aqua"
    assert public[public.index("--wandb-mode") + 1] == "offline"
    assert aqua[aqua.index("--wandb-mode") + 1] == "offline"


def test_wandb_offline_sync_orders_transactions(tmp_path) -> None:
    root = tmp_path / "wandb"
    root.mkdir()
    for timestamp in ("20260101_000001", "20260101_000002"):
        path = root / f"offline-run-{timestamp}-format-0100-prose"
        path.mkdir()
        (path / "run-format-0100-prose.wandb").write_bytes(b"transaction")
    ignored = root / "offline-run-20260101_000003-format-0100-other"
    ignored.mkdir()

    directories = sync_module.offline_run_directories(root, "format-0100-prose")
    command = sync_module.sync_command("format-0100-prose", directories)

    assert len(directories) == 2
    assert directories == sorted(directories)
    assert ignored not in directories
    assert command[command.index("--id") + 1] == "format-0100-prose"
    assert command[-1].endswith("offline-run-20260101_000002-format-0100-prose")
