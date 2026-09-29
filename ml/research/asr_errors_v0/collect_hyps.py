"""Copy reusable hypotheses into /data/phonon_asr_errors_v0/hyps/<tag>.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

from common import (
    CLIPS_PATH,
    GATE_TO_SPLIT,
    LATENT,
    LATENT_TAGS,
    OPTION0_GATES,
    OPTION0_PREDS,
    OPTION0_TAGS,
    SYSTEMS,
    V0_PARAKEET,
    V1_UNIFIED,
    append_jsonl,
    ensure_dirs,
    hyp_path,
    load_done_ids,
    read_jsonl,
)


def add(tag: str, cid: str, hypothesis: str, source: str, wanted: set[str], seen: dict[str, set[str]]) -> int:
    if cid not in wanted or cid in seen[tag]:
        return 0
    text = str(hypothesis or "").strip()
    append_jsonl(
        hyp_path(tag),
        {"id": cid, "hypothesis": text, "source": source},
    )
    seen[tag].add(cid)
    return 1


def main() -> int:
    ensure_dirs()
    clips = read_jsonl(CLIPS_PATH)
    if not clips:
        raise FileNotFoundError(CLIPS_PATH)
    wanted = {c["id"] for c in clips}
    split_of = {c["id"]: c["split"] for c in clips}
    seen: dict[str, set[str]] = {s["tag"]: load_done_ids(hyp_path(s["tag"])) for s in SYSTEMS}
    added = {tag: 0 for tag in seen}

    for clip in clips:
        if clip.get("wispr_asr"):
            added["wispr_asr"] += add(
                "wispr_asr", clip["id"], clip["wispr_asr"], "wispr_export", wanted, seen
            )

    for tag, option_tag in OPTION0_TAGS.items():
        for gate in OPTION0_GATES:
            path = OPTION0_PREDS / gate / f"{option_tag}.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                cid = row["id"]
                if split_of.get(cid) != GATE_TO_SPLIT[gate]:
                    continue
                added[tag] += add(tag, cid, row.get("hypothesis") or "", f"option0:{gate}", wanted, seen)

    for tag, folder in LATENT_TAGS.items():
        for gate, split in GATE_TO_SPLIT.items():
            directory = LATENT / folder / gate
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                cid = path.stem
                if split_of.get(cid) != split:
                    continue
                payload = json.loads(path.read_text())
                hyp = payload.get("hypothesis")
                if hyp is None:
                    continue
                added[tag] += add(tag, cid, hyp, f"latent_v0:{gate}", wanted, seen)

    if V0_PARAKEET.exists():
        for row in read_jsonl(V0_PARAKEET):
            cid = row["id"]
            added["parakeet-tdt-0.6b-v2"] += add(
                "parakeet-tdt-0.6b-v2",
                cid,
                row.get("parakeet_raw") or "",
                "parakeet_v2_wispr.jsonl",
                wanted,
                seen,
            )

    if V1_UNIFIED.exists():
        for row in read_jsonl(V1_UNIFIED):
            added["parakeet-unified-en-0.6b"] += add(
                "parakeet-unified-en-0.6b",
                row["id"],
                row.get("hypothesis") or "",
                "corrector_v1",
                wanted,
                seen,
            )

    summary = []
    for system in SYSTEMS:
        tag = system["tag"]
        have = load_done_ids(hyp_path(tag))
        missing = sorted(wanted - have)
        summary.append(
            {
                "tag": tag,
                "have": len(have & wanted),
                "missing": len(missing),
                "added_this_run": added[tag],
            }
        )
        print(
            f"{tag}: have={len(have & wanted)}/{len(wanted)} missing={len(missing)} added={added[tag]}",
            flush=True,
        )
    (Path("/data/phonon_asr_errors_v0") / "collect_summary.json").write_text(
        json.dumps({"clips": len(wanted), "systems": summary}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
