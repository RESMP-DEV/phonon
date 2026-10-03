from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "quantize_voxtral_mxfp4.py"
spec = importlib.util.spec_from_file_location("quantize_voxtral_mxfp4", MODULE_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

CONVERTER_PATH = HERE.parent / "convert_voxtral_mxfp4_vllm.py"
converter_spec = importlib.util.spec_from_file_location(
    "convert_voxtral_mxfp4_vllm", CONVERTER_PATH
)
assert converter_spec is not None
converter = importlib.util.module_from_spec(converter_spec)
sys.modules[converter_spec.name] = converter
converter_spec.loader.exec_module(converter)


def write_audio(root: Path, name: str, content: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def make_selection(tmp_path: Path) -> module.CalibrationSelection:
    audio = tmp_path / "audio"
    write_audio(audio, "a.wav", b"a")
    write_audio(audio, "b.wav", b"b")
    write_audio(audio, "eval.wav", b"eval")
    write_audio(audio, "dev.wav", b"dev")
    write_audio(audio, "short.wav", b"short")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"audio": "a.wav", "corrected": "private accepted a", "duration": 3, "split": "train"},
        {"audio": "b.wav", "ref": "private reference b", "duration": 4, "split": "train"},
        {"audio": "eval.wav", "corrected": "evaluation", "duration": 5, "split": "train"},
        {"audio": "dev.wav", "corrected": "development", "duration": 5, "split": "dev"},
        {"audio": "short.wav", "corrected": "too short", "duration": 1, "split": "train"},
        {"audio": "a.wav", "corrected": "duplicate", "duration": 6, "split": "train"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    eval_slice = tmp_path / "eval.jsonl"
    eval_slice.write_text(json.dumps({"audio": "eval.wav"}) + "\n")
    return module.select_calibration_rows(
        manifest,
        eval_slice,
        audio,
        limit=2,
        seed="stable-seed",
        min_duration=2,
        max_duration=20,
        allow_splits=["train"],
    )


def test_calibration_selection_is_stable_and_excludes_eval_and_non_train(
    tmp_path: Path,
) -> None:
    first = make_selection(tmp_path)
    assert first.stats["selected"] == 2
    assert first.stats["eval_excluded"] == 1
    assert first.stats["disallowed_split"] == 1
    assert first.stats["invalid_duration"] == 1
    assert first.stats["duplicate_audio"] == 1
    assert {row["audio"] for row in first.rows} <= {"a.wav", "b.wav"}
    assert all(row["split"] == "train" for row in first.rows)


def test_calibration_manifest_hashes_text_without_copying_it(tmp_path: Path) -> None:
    selection = make_selection(tmp_path)
    output = tmp_path / "calibration.json"
    manifest_sha = module.write_calibration_manifest(output, selection)
    text = output.read_text()
    assert "private accepted" not in text
    assert "private reference" not in text
    assert all("text_sha256" in row for row in json.loads(text)["rows"])
    assert module.sha256_file(output) == manifest_sha


def test_audio_config_patch_and_validation() -> None:
    config: dict[str, Any] = {"audio_config": {}}
    module.patch_audio_config(config)
    module.validate_audio_config(config)
    assert config["audio_config"]["hop_length"] == 160
    assert config["audio_config"]["global_log_mel_max"] is None


def test_download_contract_excludes_duplicate_consolidated_file() -> None:
    assert "model-000*.safetensors" in module.DOWNLOAD_ALLOW_PATTERNS
    assert "consolidated.safetensors" not in module.DOWNLOAD_ALLOW_PATTERNS
    assert module.EXPECTED_SHARDS == 11


def test_runtime_uses_sequential_two_gpu_plan() -> None:
    assert module.DEFAULT_DEVICE_MAP == "cpu"
    assert module.DEFAULT_PIPELINE == "sequential"
    assert module.DEFAULT_SEQUENTIAL_OFFLOAD_DEVICE == "cuda:1"


def test_vllm_weight_names_use_exact_mistral_layout() -> None:
    assert (
        converter.rename_weight("language_model.model.layers.1.mlp.up_proj.weight_packed")
        == "model.layers.1.mlp.up_proj.weight_packed"
    )
    assert converter.rename_weight("audio_tower.embed_positions.weight") is None
    assert (
        converter.rename_weight("audio_tower.conv1.weight")
        == "mm_whisper_embeddings.whisper_encoder.conv_layers.0.weight"
    )
    assert (
        converter.rename_weight("audio_tower.layer_norm.bias")
        == "mm_whisper_embeddings.whisper_encoder.transformer.norm.bias"
    )
    assert (
        converter.rename_weight("audio_tower.layers.7.self_attn.q_proj.weight")
        == "mm_whisper_embeddings.whisper_encoder.transformer.layers.7.attention.wq.weight"
    )
    assert (
        converter.rename_weight("audio_tower.layers.7.fc2.bias")
        == "mm_whisper_embeddings.whisper_encoder.transformer.layers.7.feed_forward.w2.bias"
    )
    assert (
        converter.rename_weight("multi_modal_projector.linear_1.weight")
        == "mm_whisper_embeddings.audio_language_projection.0.weight"
    )


def test_vllm_config_adds_runtime_audio_ignores() -> None:
    config = {
        **module.AUDIO_CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "ignore": ["lm_head", "re:.*audio_tower.*"],
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "type": "float",
                        "scale_dtype": "torch.uint8",
                    },
                }
            },
        },
    }
    updated = converter.update_vllm_config(config)
    assert "output" in updated["quantization_config"]["ignore"]
    assert "language_model.lm_head" in updated["quantization_config"]["ignore"]
    assert "re:.*whisper_encoder.*" in updated["quantization_config"]["ignore"]
    assert "re:.*audio_language_adapter.*" in updated["quantization_config"]["ignore"]


def test_output_audit_rejects_quantized_audio_or_lm_head() -> None:
    packed = {
        "model.layers.0.self_attn.q_proj.weight_packed": ("uint8", [8, 16]),
        "model.audio_tower.layers.0.fc1.weight_packed": ("uint8", [8, 16]),
        "lm_head.weight_packed": ("uint8", [8, 16]),
    }
    scales = {
        "model.layers.0.self_attn.q_proj.weight_scale": ("uint8", [8, 1]),
    }
    try:
        module.validate_safetensors_header(packed, scales)
    except ValueError as error:
        assert "ignored model component" in str(error)
    else:
        raise AssertionError("expected quantized audio tower and lm_head to fail")


def test_output_config_requires_mxfp4_compressed_tensors() -> None:
    config = {
        **module.AUDIO_CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "type": "float",
                        "scale_dtype": "torch.uint8",
                    },
                }
            },
        },
    }
    audit = module.validate_output_config(config)
    assert audit["format"] == "mxfp4-pack-quantized"


def test_output_config_rejects_int4_gptq_as_mxfp4() -> None:
    config = {
        **module.AUDIO_CONFIG,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "num_bits": 4,
                    "group_size": 128,
                    "weights": {
                        "num_bits": 4,
                        "group_size": 128,
                        "type": "int",
                        "scale_dtype": None,
                    },
                }
            },
        },
    }
    try:
        module.validate_output_config(config)
    except ValueError as error:
        assert "mxfp4-pack-quantized" in str(error)
    else:
        raise AssertionError("expected INT4 GPTQ config to fail")
