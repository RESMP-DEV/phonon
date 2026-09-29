"""Build the 804-clip eval table with references. Writes /data/phonon_asr_errors_v0/clips.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from common import (
    CLIPS_PATH,
    JULY_ROOT,
    LATENT,
    OPTION0,
    REPO,
    WISPR_ROOT,
    ensure_dirs,
    write_jsonl,
)

sys.path.insert(0, str(OPTION0))
sys.path.insert(0, str(REPO / "src"))


def wispr_reference(row: dict) -> str:
    edited = str(row.get("editedText") or "").strip()
    formatted = str(row.get("formattedText") or "").strip()
    return edited if edited else formatted


def build() -> list[dict]:
    from gates import load_gate

    holdout_ids = {
        json.loads(line)["transcriptEntityId"]
        for line in (WISPR_ROOT / "wispr_holdout120.jsonl").read_text().splitlines()
        if line.strip()
    }
    clips: list[dict] = []
    seen: set[str] = set()
    for row in [
        json.loads(line)
        for line in (WISPR_ROOT / "manifest_with_audio.jsonl").read_text().splitlines()
        if line.strip()
    ]:
        cid = row["transcriptEntityId"]
        if cid in seen:
            raise ValueError(f"duplicate wispr id {cid}")
        seen.add(cid)
        split = "holdout120" if cid in holdout_ids else "wispr_train640"
        wav = WISPR_ROOT / "audio" / f"{cid}.wav"
        if not wav.is_file():
            wav = WISPR_ROOT / str(row.get("audio_path") or "")
        ref = wispr_reference(row)
        clips.append(
            {
                "id": cid,
                "split": split,
                "wav": str(wav),
                "reference": ref,
                "has_reference": bool(ref),
                "wispr_asr": str(row.get("asrText") or "").strip(),
                "formatted_text": str(row.get("formattedText") or "").strip(),
                "edited_text": str(row.get("editedText") or "").strip(),
                "duration": float(row.get("wav_seconds") or row.get("duration") or 0.0),
                "source_set": "wispr_20260915",
                "wispr_status": row.get("status"),
            }
        )

    edit_gate = {row["id"]: row for row in load_gate("wispr_edit25")}
    samples = json.loads((JULY_ROOT / "edit_samples.json").read_text())
    for sample in samples:
        cid = sample["id"]
        if cid in seen:
            raise ValueError(f"id collision {cid}")
        seen.add(cid)
        wav = JULY_ROOT / "audio" / Path(sample["audio_wav"]).name
        ref = str(sample.get("edited") or "").strip()
        gate_ref = str(edit_gate[cid]["reference"]).strip()
        if ref != gate_ref:
            ref = gate_ref
        clips.append(
            {
                "id": cid,
                "split": "edit25",
                "wav": str(wav),
                "reference": ref,
                "has_reference": bool(ref),
                "wispr_asr": str(sample.get("asr") or "").strip(),
                "formatted_text": str(sample.get("formatted") or "").strip(),
                "edited_text": ref,
                "duration": float(sample.get("duration") or 0.0),
                "source_set": "dictation_eval_20260719",
            }
        )

    for row in load_gate("aqua_new_holdout"):
        cid = row["id"]
        if cid in seen:
            raise ValueError(f"id collision {cid}")
        seen.add(cid)
        wav = LATENT / "audio" / "aqua_new_holdout" / f"{cid}.wav"
        if not wav.is_file():
            raise FileNotFoundError(wav)
        clips.append(
            {
                "id": cid,
                "split": "aqua19",
                "wav": str(wav),
                "reference": row["reference"],
                "has_reference": bool(row["reference"]),
                "wispr_asr": "",
                "formatted_text": "",
                "edited_text": "",
                "duration": 0.0,
                "source_set": "aqua_new_holdout",
            }
        )

    missing = [c for c in clips if not Path(c["wav"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} wavs, first={missing[0]}")
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip["split"]] = counts.get(clip["split"], 0) + 1
    if counts.get("holdout120") != 120:
        raise ValueError(counts)
    if counts.get("wispr_train640") != 640:
        raise ValueError(counts)
    if counts.get("edit25") != 25:
        raise ValueError(counts)
    if counts.get("aqua19") != 19:
        raise ValueError(counts)
    return clips


def main() -> int:
    ensure_dirs()
    clips = build()
    write_jsonl(CLIPS_PATH, clips)
    print(json.dumps({"clips": len(clips), "path": str(CLIPS_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
