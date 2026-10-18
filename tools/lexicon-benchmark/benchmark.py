#!/usr/bin/env python3
"""Rank technical lexicon terms by measured dictation difficulty.

The difficulty signal is precomputed per term in oracle result files: each
entry was synthesized in two voices and transcribed, so the worst per-voice
diff class (same, case, format, phonetic) is a measured ASR risk, not a guess.
This tool aggregates that signal per origin and category and emits a ranked
report. It never reads user dictation, network, or the Aqua cache.

The ranked term list stays local: the default report writes only aggregate
counts per category and per diff class. Use --emit-terms to also write the term
list, which is still local, to an explicit path.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from collect_local import is_safe_output_path, non_negative

DIFF_ORDER = ("same", "case", "format", "phonetic")
DIFF_WEIGHT = {"same": 0.0, "case": 0.2, "format": 0.6, "phonetic": 1.0}
LEXICON_DIR = Path(__file__).resolve().parent.parent / "profile-miner" / "lexicon"
SCHEMA_VERSION = 2


def diff_class(term: str, heard: str) -> str:
    term_stripped = term.strip()
    heard_stripped = heard.strip().rstrip(".!?,")
    if heard_stripped == term_stripped:
        return "same"
    if heard_stripped.lower() == term_stripped.lower():
        return "case"
    if reformat_key(heard_stripped) == reformat_key(term_stripped):
        return "format"
    return "phonetic"


def reformat_key(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def worst_class(term: str, voices: dict[str, Any]) -> str:
    worst = "same"
    for heard in voices.values():
        if heard is None:
            continue
        current = diff_class(term, str(heard))
        if DIFF_ORDER.index(current) > DIFF_ORDER.index(worst):
            worst = current
    return worst


def load_lexicon(path: Path | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        with gzip.open(
            LEXICON_DIR / "lexicon.json.gz", "rt", encoding="utf-8"
        ) as handle:
            return json.load(handle)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_spoken(path: Path | None = None) -> dict[str, dict[str, Any]]:
    spoken: dict[str, dict[str, Any]] = {}
    spoken_path = path or LEXICON_DIR / "spoken.jsonl.gz"
    if not spoken_path.exists():
        if path is not None:
            raise FileNotFoundError(f"spoken results not found: {path}")
        return spoken
    if path is None:
        with gzip.open(spoken_path, "rt", encoding="utf-8") as handle:
            lines = handle.readlines()
    else:
        with spoken_path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        spoken[str(row["term"])] = row.get("voices", {})
    return spoken


def build_rows(
    lexicon: dict[str, dict[str, Any]],
    spoken: dict[str, dict[str, Any]],
    origin: str = "shipped",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in lexicon.values():
        term = str(entry["term"])
        voices = spoken.get(term, {})
        heard_values = [value for value in voices.values() if value is not None]
        if not heard_values:
            continue
        klass = worst_class(term, voices)
        rows.append(
            {
                "term": term,
                "cats": entry.get("cats", []),
                "diff_class": klass,
                "risk": DIFF_WEIGHT[klass],
                "origin": origin,
                "priority": float(entry.get("priority", 0.5)),
                "voice_count": len(heard_values),
                "spoken_forms": sorted({str(value) for value in heard_values}),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, int] = {name: 0 for name in DIFF_ORDER}
    by_origin: dict[str, int] = defaultdict(int)
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in DIFF_ORDER}
    )
    category_totals: dict[str, int] = defaultdict(int)
    complete_rows = [row for row in rows if row.get("voice_count", 2) >= 2]
    incomplete_voice_terms = len(rows) - len(complete_rows)
    for row in complete_rows:
        by_class[row["diff_class"]] += 1
        by_origin[row["origin"]] += 1
        for category in row["cats"] or ["uncategorized"]:
            by_category[category][row["diff_class"]] += 1
            category_totals[category] += 1
    risky_categories = []
    for category, counts in by_category.items():
        total = category_totals[category]
        risky = counts["phonetic"] + counts["format"]
        risky_categories.append(
            {
                "category": category,
                "terms": total,
                "risky_terms": risky,
                "risky_share": round(risky / total, 4) if total else 0.0,
                **{f"class_{name}": counts[name] for name in DIFF_ORDER},
            }
        )
    risky_categories.sort(
        key=lambda item: (-item["risky_share"], -item["risky_terms"], item["category"])
    )
    return {
        "terms": len(rows),
        "complete_voice_terms": len(complete_rows),
        "incomplete_voice_terms": incomplete_voice_terms,
        "by_diff_class": by_class,
        "by_origin": dict(sorted(by_origin.items())),
        "multi_label_terms": sum(
            bool(row["cats"] and len(row["cats"]) > 1) for row in rows
        ),
        "risky_terms": by_class["phonetic"] + by_class["format"],
        "risky_share": round(
            (by_class["phonetic"] + by_class["format"]) / len(complete_rows), 4
        )
        if complete_rows
        else 0.0,
        "categories": risky_categories,
    }


def stratify(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Round-robin ASR class and category within one origin."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in row["cats"] or ["uncategorized"]:
            groups[(row["diff_class"], row["origin"], category)].append(row)
    ordered_groups = sorted(
        (
            sorted(group, key=lambda row: (-row["risk"], -row["priority"], row["term"]))
            for group in groups.values()
        ),
        key=lambda group: (
            -DIFF_ORDER.index(group[0]["diff_class"]),
            -group[0]["priority"],
            group[0]["origin"],
            group[0]["cats"][0] if group[0]["cats"] else "uncategorized",
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < size:
        added = False
        for group in ordered_groups:
            while group:
                candidate = group.pop(0)
                if candidate["term"] not in seen:
                    selected.append(candidate)
                    seen.add(candidate["term"])
                    added = True
                    break
            if len(selected) >= size:
                break
        if not added:
            break
    return selected[:size]


def stratified_slice(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Balance origin first, then ASR class and category within each origin."""
    origins = sorted(
        {row["origin"] for row in rows},
        key=lambda origin: (origin != "local-inventory", origin),
    )
    if not origins:
        return []
    base_quota, remainder = divmod(size, len(origins))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, origin in enumerate(origins):
        quota = base_quota + (1 if index < remainder else 0)
        origin_rows = [row for row in rows if row["origin"] == origin]
        for row in stratify(origin_rows, quota):
            if row["term"] not in seen:
                selected.append(row)
                seen.add(row["term"])
    if len(selected) < size:
        remaining_rows = [row for row in rows if row["term"] not in seen]
        for row in stratify(remaining_rows, size - len(selected)):
            if row["term"] not in seen:
                selected.append(row)
                seen.add(row["term"])
    return selected[:size]


def public_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Project only aggregate collector fields into a shareable report."""
    allowed = (
        "schema_version",
        "summary",
        "sources",
        "collection_failures",
        "non_claims",
    )
    return {key: value for key, value in data.items() if key in allowed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emit-terms", type=Path, default=None)
    parser.add_argument("--emit-stratified", type=Path, default=None)
    parser.add_argument("--stratified-size", type=non_negative, default=200)
    parser.add_argument("--extra-lexicon", type=Path, default=None)
    parser.add_argument("--extra-spoken", type=Path, default=None)
    parser.add_argument("--extra-manifest", type=Path, default=None)
    parser.add_argument(
        "--top",
        type=non_negative,
        default=0,
        help="0 means every term in the term export",
    )
    parser.add_argument("--limit", type=non_negative, default=0)
    arguments = parser.parse_args(argv)

    for export in (arguments.emit_terms, arguments.emit_stratified):
        if export is not None and not is_safe_output_path(export):
            raise ValueError(
                "term exports carry local inventory rows and spoken forms; they must "
                f"stay outside the repository or under ignored build/: {export}"
            )
    output_paths = [
        arguments.out,
        *(path for path in (arguments.emit_terms, arguments.emit_stratified) if path),
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("benchmark output paths must be distinct")

    extension_flags = (
        arguments.extra_lexicon,
        arguments.extra_spoken,
        arguments.extra_manifest,
    )
    if any(extension_flags) != all(extension_flags):
        raise ValueError(
            "--extra-lexicon, --extra-spoken, and --extra-manifest must be used together"
        )

    rows = build_rows(load_lexicon(), load_spoken())
    sources = ["shipped public lexicon"]
    if arguments.extra_lexicon and arguments.extra_spoken:
        rows.extend(
            build_rows(
                load_lexicon(arguments.extra_lexicon),
                load_spoken(arguments.extra_spoken),
                origin="local-inventory",
            )
        )
        sources.append("local tool and registry inventory")
    rows.sort(key=lambda row: (-row["risk"], -row["priority"], row["term"]))
    if arguments.limit:
        rows = rows[: arguments.limit]
    summary = summarize(rows)
    extension_manifest = None
    if arguments.extra_manifest:
        extension_manifest = public_manifest(
            json.loads(arguments.extra_manifest.read_text(encoding="utf-8"))
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "signal": "worst per-voice ASR diff class over two synthesized voices",
        "summary": summary,
        "local_inventory": extension_manifest,
        "non_claims": [
            "This ranks public lexicon and local technical inventory terms; no user dictation is measured.",
            "Risk is measured with one ASR model on synthesized speech, so it is a triage signal rather than a product WER claim.",
            "A high risk term can still be corrected by the dictionary; ranking does not prove product impact.",
        ],
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out": str(arguments.out),
                "terms": summary["terms"],
                "risky_share": summary["risky_share"],
            }
        )
    )

    if arguments.emit_terms:
        exported = rows[: arguments.top] if arguments.top else rows
        arguments.emit_terms.parent.mkdir(parents=True, exist_ok=True)
        arguments.emit_terms.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported),
            encoding="utf-8",
        )
        print(
            json.dumps({"emit_terms": str(arguments.emit_terms), "rows": len(exported)})
        )
    if arguments.emit_stratified:
        complete_rows = [row for row in rows if row.get("voice_count", 2) >= 2]
        exported = stratified_slice(complete_rows, arguments.stratified_size)
        arguments.emit_stratified.parent.mkdir(parents=True, exist_ok=True)
        arguments.emit_stratified.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "emit_stratified": str(arguments.emit_stratified),
                    "rows": len(exported),
                }
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
