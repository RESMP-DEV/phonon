from __future__ import annotations

import csv
import gc
import json
import mimetypes
import os
import re
import subprocess
import tarfile
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from phonon.audio import write_audio_bytes
from phonon.metrics import (
    casing_error_rate,
    compute_word_wer_cer,
    compute_wer_cer,
    hallucination_omission_counts,
    punctuation_error_rate,
    technical_term_error_rate,
)
from phonon.exclusions import filter_excluded_rows
from phonon.schema import read_parquet_rows


PREDICTION_TEXT_FIELDS = ("hypothesis", "text", "transcript", "prediction", "output")


def gpu_memory_mib() -> int | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
        return int(out)
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def _audio_temp_files(rows: list[dict[str, Any]], tmpdir: Path) -> list[str]:
    paths: list[str] = []
    for row in rows:
        audio = row.get("audio") or {}
        audio_name = audio.get("path") or f"{row['id'].replace(':', '_')}.wav"
        path = tmpdir / audio_name
        write_audio_bytes(audio, path)
        paths.append(str(path))
    return paths


def _is_canary_qwen(model_name: str) -> bool:
    return "canary-qwen" in model_name.lower()


def _is_canary_multitask(model_name: str) -> bool:
    model = model_name.lower()
    return "canary" in model and "canary-qwen" not in model


def _is_parakeet_unified(model_name: str) -> bool:
    return "parakeet-unified-en-0.6b" in model_name.lower()


def _nemo_transcribe_kwargs(model_name: str) -> dict[str, str]:
    if _is_canary_multitask(model_name):
        return {"source_lang": "en", "target_lang": "en", "pnc": "yes"}
    return {}


def _decode_nemo_output(output: Any) -> str:
    return str(getattr(output, "text", output)).strip()


def _nemo_output_token_count(output: Any) -> int | None:
    y_sequence = getattr(output, "y_sequence", None)
    if y_sequence is None:
        return None
    try:
        return int(len(y_sequence))
    except TypeError:
        return None


def _nemo_output_metadata(output: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    score = getattr(output, "score", None)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None
    if score_value is not None:
        metadata["score"] = score_value
    token_count = _nemo_output_token_count(output)
    if token_count is not None:
        metadata["token_count"] = token_count
    if score_value is not None and token_count:
        metadata["score_per_token"] = score_value / token_count
    return metadata


def _patch_parakeet_unified_offline_config(config: Any) -> Any:
    if "att_chunk_context_size" in config.encoder:
        del config.encoder["att_chunk_context_size"]
    if "conv_context_style" in config.encoder:
        del config.encoder["conv_context_style"]
    config.encoder.att_context_style = "regular"
    if not config.get("validation_ds"):
        config.validation_ds = {"use_start_end_token": False}
    return config


def restore_parakeet_unified(nemo_asr: Any, model_name: str) -> Any:
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    nemo_path = hf_hub_download(repo_id=model_name, filename="parakeet-unified-en-0.6b.nemo")
    with tempfile.TemporaryDirectory(prefix="phonon-nemo-config-") as tmp:
        config_path = Path(tmp) / "model_config.yaml"
        with tarfile.open(nemo_path) as archive:
            config_file = archive.extractfile("./model_config.yaml")
            if config_file is None:
                raise RuntimeError(f"{model_name} checkpoint did not include model_config.yaml")
            config_path.write_text(config_file.read().decode())
        config = _patch_parakeet_unified_offline_config(OmegaConf.load(config_path))
        OmegaConf.save(config, config_path)
        return nemo_asr.models.ASRModel.restore_from(
            nemo_path,
            override_config_path=str(config_path),
        )


def transcribe_nemo(model_name: str, audio_files: list[str], batch_size: int) -> tuple[list[str], float]:
    records, model_load_seconds = transcribe_nemo_detailed(model_name, audio_files, batch_size)
    return [str(record["text"]) for record in records], model_load_seconds


def transcribe_nemo_detailed(
    model_name: str,
    audio_files: list[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    if _is_canary_qwen(model_name):
        texts, model_load_seconds = transcribe_canary_qwen(model_name, audio_files, batch_size)
        return [{"text": text} for text in texts], model_load_seconds

    import nemo.collections.asr as nemo_asr

    t0 = time.perf_counter()
    if model_name.endswith(".nemo") or Path(model_name).exists():
        model = nemo_asr.models.ASRModel.restore_from(
            model_name,
            save_restore_connector=_trusted_local_nemo_connector(),
        )
    elif _is_parakeet_unified(model_name):
        model = restore_parakeet_unified(nemo_asr, model_name)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model_load_seconds = time.perf_counter() - t0
    outputs = model.transcribe(
        audio_files,
        batch_size=batch_size,
        return_hypotheses=True,
        **_nemo_transcribe_kwargs(model_name),
    )
    records = [
        {
            "text": _decode_nemo_output(output),
            **_nemo_output_metadata(output),
        }
        for output in outputs
    ]
    del model
    gc.collect()
    return records, model_load_seconds


def transcribe_canary_qwen(
    model_name: str,
    audio_files: list[str],
    batch_size: int,
) -> tuple[list[str], float]:
    import torch
    from nemo.collections.speechlm2.models import SALM

    t0 = time.perf_counter()
    model = SALM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    model_load_seconds = time.perf_counter() - t0
    hypotheses: list[str] = []
    effective_batch_size = max(1, batch_size)
    with torch.inference_mode():
        for start in range(0, len(audio_files), effective_batch_size):
            batch = audio_files[start : start + effective_batch_size]
            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [audio_file],
                    }
                ]
                for audio_file in batch
            ]
            answer_ids = model.generate(prompts=prompts, max_new_tokens=512)
            for token_ids in answer_ids:
                if hasattr(token_ids, "cpu"):
                    token_ids = token_ids.cpu()
                hypotheses.append(str(model.tokenizer.ids_to_text(token_ids)).strip())
    del model
    gc.collect()
    return hypotheses, model_load_seconds


def _trusted_local_nemo_connector() -> Any:
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    class TrustedLocalConnector(SaveRestoreConnector):
        def _load_state_dict_from_disk(self, model_weights: str, map_location: Any = None) -> Any:
            import torch

            return torch.load(model_weights, map_location=map_location, weights_only=False)

    return TrustedLocalConnector()


def transcribe_whisper(model_name: str, audio_files: list[str]) -> tuple[list[str], float]:
    import whisper

    t0 = time.perf_counter()
    model = whisper.load_model(model_name)
    model_load_seconds = time.perf_counter() - t0
    hypotheses: list[str] = []
    for audio_file in audio_files:
        result = model.transcribe(audio_file, fp16=True, verbose=False)
        hypotheses.append((result.get("text") or "").strip())
    del model
    gc.collect()
    return hypotheses, model_load_seconds


def _coerce_transformers_text(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("text") or "").strip()
    return str(output).strip()


def _decode_transformers_batch(
    processor: Any,
    outputs: Any,
    audio_chunk_index: Any,
) -> list[str]:
    decode_kwargs = {"skip_special_tokens": True}
    if audio_chunk_index is not None:
        decode_kwargs["audio_chunk_index"] = audio_chunk_index
    try:
        decoded = processor.decode(outputs, language="en", **decode_kwargs)
    except TypeError:
        decoded = processor.decode(outputs, **decode_kwargs)
    if isinstance(decoded, str):
        return [decoded.strip()]
    return [str(item).strip() for item in decoded]


def transcribe_transformers_asr(
    model_name: str,
    audio_files: list[str],
    batch_size: int,
) -> tuple[list[str], float]:
    import torch
    from transformers import AutoProcessor, pipeline
    from transformers.audio_utils import load_audio

    t0 = time.perf_counter()
    try:
        from transformers import CohereAsrForConditionalGeneration
    except ImportError:
        CohereAsrForConditionalGeneration = None

    if CohereAsrForConditionalGeneration and "cohere-transcribe" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name)
        model = CohereAsrForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        model.eval()
        model_load_seconds = time.perf_counter() - t0
        hypotheses: list[str] = []
        effective_batch_size = max(1, batch_size)
        with torch.inference_mode():
            for start in range(0, len(audio_files), effective_batch_size):
                batch_files = audio_files[start : start + effective_batch_size]
                audio_batch = [load_audio(path, sampling_rate=16000) for path in batch_files]
                inputs = processor(
                    audio_batch,
                    sampling_rate=16000,
                    return_tensors="pt",
                    language="en",
                )
                audio_chunk_index = inputs.get("audio_chunk_index")
                inputs.to(model.device, dtype=model.dtype)
                outputs = model.generate(**inputs, max_new_tokens=512)
                hypotheses.extend(_decode_transformers_batch(processor, outputs, audio_chunk_index))
        del model
        gc.collect()
        return hypotheses, model_load_seconds

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        trust_remote_code=True,
        device=device,
    )
    model_load_seconds = time.perf_counter() - t0
    raw_outputs = pipe(audio_files, batch_size=max(1, batch_size))
    if isinstance(raw_outputs, dict):
        raw_outputs = [raw_outputs]
    hypotheses = [_coerce_transformers_text(output) for output in raw_outputs]
    del pipe
    gc.collect()
    return hypotheses, model_load_seconds


def _multipart_form_data(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    boundary: str,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks)


def transcribe_openai_compatible(
    model_name: str,
    audio_files: list[str],
    base_url: str,
    api_key_env: str,
    request_timeout: float = 120.0,
) -> tuple[list[str], float]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")
    endpoint = f"{base_url.rstrip('/')}/audio/transcriptions"
    hypotheses: list[str] = []
    for audio_file in audio_files:
        boundary = f"phonon-{uuid.uuid4().hex}"
        body = _multipart_form_data(
            {"model": model_name, "response_format": "json"},
            "file",
            Path(audio_file),
            boundary,
        )
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"transcription API failed with HTTP {exc.code}: {detail}") from exc
        text = payload.get("text")
        if text is None:
            raise RuntimeError(f"transcription API response did not include text: {payload}")
        hypotheses.append(str(text).strip())
    return hypotheses, 0.0


def _prediction_rows(rows: list[dict[str, Any]], hypotheses: list[str]) -> list[dict[str, Any]]:
    prediction_rows = []
    for row, hyp in zip(rows, hypotheses, strict=True):
        prediction_rows.append(
            {
                "id": row["id"],
                "sha256_audio": row["sha256_audio"],
                "split": row["split"],
                "source_kind": row.get("source_kind") or "",
                "source_id": row.get("source_id") or "",
                "clip_start": row.get("clip_start"),
                "clip_end": row.get("clip_end"),
                "duration_seconds": row["duration_seconds"],
                "label_status": row.get("label_status") or "",
                "domain_tags": row.get("domain_tags") or [],
                "verbatim_text": row.get("verbatim_text") or "",
                "insert_text": row.get("insert_text") or "",
                "hypothesis": hyp,
                "term_spans": row.get("term_spans") or [],
                "difficulty_tags": row.get("difficulty_tags") or [],
            }
        )
    return prediction_rows


def _write_predictions_and_metrics(
    *,
    adapter: str,
    model: str,
    dataset: str,
    split: str,
    rows: list[dict[str, Any]],
    hypotheses: list[str],
    out: Path,
    elapsed_seconds: float | None,
    model_load_seconds: float | None,
    batch_size: int | None,
    gpu_before: int | None = None,
    gpu_after: int | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in _prediction_rows(rows, hypotheses):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    audio_seconds = sum(float(row["duration_seconds"]) for row in rows)
    verbatim_refs = [row.get("verbatim_text") or "" for row in rows]
    insert_refs = [row.get("insert_text") or row.get("normalized_text") or "" for row in rows]
    term_lists = [row.get("term_spans") or [] for row in rows]

    metrics: dict[str, Any] = {
        "adapter": adapter,
        "model": model,
        "dataset": dataset,
        "split": split,
        "utterances": len(rows),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "model_load_seconds": model_load_seconds,
        "realtime_factor": elapsed_seconds / audio_seconds if elapsed_seconds and audio_seconds else None,
        "throughput_x_realtime": audio_seconds / elapsed_seconds if elapsed_seconds else None,
        "batch_size": batch_size,
        "gpu_memory_mib_before": gpu_before,
        "gpu_memory_mib_after": gpu_after,
        "output_jsonl": str(out),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics.update({f"verbatim_{k}": v for k, v in compute_wer_cer(verbatim_refs, hypotheses).items()})
    metrics.update({f"insert_{k}": v for k, v in compute_wer_cer(insert_refs, hypotheses).items()})
    metrics.update(
        {f"verbatim_word_{k}": v for k, v in compute_word_wer_cer(verbatim_refs, hypotheses).items()}
    )
    metrics.update(
        {f"insert_word_{k}": v for k, v in compute_word_wer_cer(insert_refs, hypotheses).items()}
    )
    metrics.update(technical_term_error_rate(term_lists, hypotheses))
    metrics["insert_casing_error_rate"] = casing_error_rate(insert_refs, hypotheses)
    metrics["insert_punctuation_error_rate"] = punctuation_error_rate(insert_refs, hypotheses)
    metrics.update(hallucination_omission_counts(insert_refs, hypotheses))
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _prediction_text(row: dict[str, Any]) -> str:
    for field in PREDICTION_TEXT_FIELDS:
        value = row.get(field)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"prediction row for {row.get('id')!r} has no text field")


def load_external_predictions(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    predictions: dict[str, str] = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row_id = (row.get("id") or "").strip()
                if row_id:
                    predictions[row_id] = _prediction_text(row)
    else:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                raise ValueError(f"prediction row {line_number} has no id")
            predictions[row_id] = _prediction_text(row)
    return predictions


def evaluate_external_predictions(
    *,
    model: str,
    predictions: Path,
    dataset_root: Path,
    dataset: str,
    split: str,
    out: Path,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_parquet_rows(dataset_root / "datasets" / dataset, split, include_audio=False)
    rows, excluded_rows = filter_excluded_rows(rows, dataset_root)
    rows = slice_eval_rows(rows, offset=offset, limit=limit)
    by_id = load_external_predictions(predictions)
    missing = [row["id"] for row in rows if row["id"] not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"external predictions missing {len(missing)} rows: {preview}")
    hypotheses = [by_id[row["id"]] for row in rows]
    return _write_predictions_and_metrics(
        adapter="external",
        model=model,
        dataset=dataset,
        split=split,
        rows=rows,
        hypotheses=hypotheses,
        out=out,
        elapsed_seconds=None,
        model_load_seconds=None,
        batch_size=None,
        extra_metrics={
            "external_predictions": str(predictions),
            "row_offset": offset,
            "row_limit": limit,
            "base_split": split,
            "excluded_rows": excluded_rows,
        },
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "row"


def export_eval_pack(
    *,
    dataset_root: Path,
    dataset: str,
    split: str,
    out_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_parquet_rows(dataset_root / "datasets" / dataset, split)
    rows, excluded_rows = filter_excluded_rows(rows, dataset_root)
    if limit is not None:
        rows = rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    template_path = out_dir / "predictions_template.csv"

    manifest_rows = []
    for index, row in enumerate(rows):
        audio_path = audio_dir / f"{index:05d}_{_safe_filename(row['id'])}.wav"
        write_audio_bytes(row.get("audio") or {}, audio_path)
        manifest_rows.append(
            {
                "id": row["id"],
                "audio_path": str(audio_path),
                "sha256_audio": row["sha256_audio"],
                "duration_seconds": row["duration_seconds"],
                "source_kind": row.get("source_kind") or "",
                "source_id": row.get("source_id") or "",
                "clip_start": row.get("clip_start"),
                "clip_end": row.get("clip_end"),
                "split": row["split"],
                "domain_tags": row.get("domain_tags") or [],
                "difficulty_tags": row.get("difficulty_tags") or [],
            }
        )

    with manifest_path.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with template_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "hypothesis"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({"id": row["id"], "hypothesis": ""})

    return {
        "dataset": dataset,
        "split": split,
        "rows": len(rows),
        "excluded_rows": excluded_rows,
        "audio_dir": str(audio_dir),
        "manifest": str(manifest_path),
        "predictions_template": str(template_path),
    }


def slice_eval_rows(
    rows: list[dict[str, Any]],
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    sliced = rows[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def evaluate_dataset(
    adapter: str,
    model: str,
    dataset_root: Path,
    dataset: str,
    split: str,
    out: Path,
    batch_size: int = 4,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    request_timeout: float = 120.0,
    predictions: Path | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    if adapter == "external":
        if predictions is None:
            raise ValueError("--predictions is required for the external adapter")
        return evaluate_external_predictions(
            model=model,
            predictions=predictions,
            dataset_root=dataset_root,
            dataset=dataset,
            split=split,
            out=out,
            offset=offset,
            limit=limit,
        )

    rows = read_parquet_rows(dataset_root / "datasets" / dataset, split)
    rows, excluded_rows = filter_excluded_rows(rows, dataset_root)
    rows = slice_eval_rows(rows, offset=offset, limit=limit)
    if not rows:
        raise ValueError(f"split {split!r} is empty")

    gpu_before = gpu_memory_mib()
    with tempfile.TemporaryDirectory(prefix="phonon-audio-") as tmp:
        audio_files = _audio_temp_files(rows, Path(tmp))
        transcribe_start = time.perf_counter()
        if adapter == "nemo":
            hypotheses, model_load_seconds = transcribe_nemo(model, audio_files, batch_size)
        elif adapter == "whisper":
            hypotheses, model_load_seconds = transcribe_whisper(model, audio_files)
        elif adapter == "transformers-asr":
            hypotheses, model_load_seconds = transcribe_transformers_asr(model, audio_files, batch_size)
        elif adapter == "openai-compatible":
            if not base_url:
                raise ValueError("--base-url is required for the openai-compatible adapter")
            hypotheses, model_load_seconds = transcribe_openai_compatible(
                model,
                audio_files,
                base_url,
                api_key_env,
                request_timeout=request_timeout,
            )
        else:
            raise ValueError(f"unknown adapter {adapter!r}")
        elapsed_seconds = time.perf_counter() - transcribe_start

    gpu_after = gpu_memory_mib()
    extra = {
        "row_offset": offset,
        "row_limit": limit,
        "base_split": split,
        "excluded_rows": excluded_rows,
    }
    if adapter == "openai-compatible":
        extra.update({"base_url": base_url, "api_key_env": api_key_env})
    return _write_predictions_and_metrics(
        adapter=adapter,
        model=model,
        dataset=dataset,
        split=split,
        rows=rows,
        hypotheses=hypotheses,
        out=out,
        elapsed_seconds=elapsed_seconds,
        model_load_seconds=model_load_seconds,
        batch_size=batch_size,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
        extra_metrics=extra,
    )
