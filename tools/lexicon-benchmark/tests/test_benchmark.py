from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import pytest
from benchmark import (
    build_rows,
    diff_class,
    main,
    public_manifest,
    stratified_slice,
    summarize,
    worst_class,
)


def test_diff_class_orders_format_before_phonetic() -> None:
    assert diff_class("sm_90", "sm 90") == "format"
    assert diff_class("cuDNN", "coulden") == "phonetic"
    assert diff_class("H100", "h100") == "case"
    assert diff_class("H100", "H100") == "same"


def test_worst_class_takes_the_worst_voice() -> None:
    assert worst_class("sm_90", {"a": "sm 90", "b": "sm ninety"}) == "phonetic"
    assert worst_class("sm_90", {"a": "SM_90"}) == "case"


def test_report_writes_aggregates_without_terms(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    emit = tmp_path / "terms.jsonl"

    assert main(["--out", str(out), "--emit-terms", str(emit), "--limit", "50"]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["terms"] > 0
    assert set(report["summary"]["by_diff_class"]) == {
        "same",
        "case",
        "format",
        "phonetic",
    }
    assert report["non_claims"]
    # The aggregate report must not embed the term list; that stays local.
    assert "spoken_forms" not in out.read_text(encoding="utf-8")
    assert emit.read_text(encoding="utf-8").count("\n") > 0


def test_summary_prefers_high_risk_categories() -> None:
    rows = [
        {
            "term": "a",
            "cats": ["gpu"],
            "diff_class": "phonetic",
            "risk": 1.0,
            "origin": "shipped",
            "priority": 0.5,
            "voice_count": 2,
            "spoken_forms": [],
        },
        {
            "term": "b",
            "cats": ["gpu"],
            "diff_class": "format",
            "risk": 0.6,
            "origin": "shipped",
            "priority": 0.5,
            "voice_count": 2,
            "spoken_forms": [],
        },
        {
            "term": "c",
            "cats": ["brew"],
            "diff_class": "same",
            "risk": 0.0,
            "origin": "shipped",
            "priority": 0.5,
            "voice_count": 2,
            "spoken_forms": [],
        },
    ]
    summary = summarize(rows)
    assert summary["risky_terms"] == 2
    assert summary["incomplete_voice_terms"] == 0
    assert summary["categories"][0]["category"] == "gpu"


def test_build_rows_skips_terms_without_oracle_evidence() -> None:
    lexicon = {"a": {"term": "A", "cats": ["gpu"]}, "b": {"term": "B", "cats": ["gpu"]}}
    spoken = {"A": {"v1": "A"}, "B": {"v1": None}}
    rows = build_rows(lexicon, spoken)
    assert [row["term"] for row in rows] == ["A"]
    assert rows[0]["diff_class"] == "same"
    assert rows[0]["origin"] == "shipped"
    assert rows[0]["voice_count"] == 1


def test_incomplete_voice_rows_do_not_enter_risk_counts() -> None:
    rows = [
        {
            "term": "complete",
            "cats": ["toolbelt"],
            "diff_class": "phonetic",
            "risk": 1.0,
            "origin": "local-inventory",
            "priority": 1.0,
            "voice_count": 2,
            "spoken_forms": [],
        },
        {
            "term": "incomplete",
            "cats": ["toolbelt"],
            "diff_class": "same",
            "risk": 0.0,
            "origin": "local-inventory",
            "priority": 1.0,
            "voice_count": 1,
            "spoken_forms": [],
        },
    ]
    summary = summarize(rows)
    assert summary["terms"] == 2
    assert summary["complete_voice_terms"] == 1
    assert summary["incomplete_voice_terms"] == 1
    assert summary["by_diff_class"]["same"] == 0
    assert summary["risky_share"] == 1.0


def test_limit_is_applied_after_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexicon_dir = tmp_path / "shipped"
    lexicon_dir.mkdir()
    rows = {
        "same": {"term": "same", "cats": ["gpu"]},
        "phonetic": {"term": "phonetic", "cats": ["gpu"]},
    }
    with gzip.open(lexicon_dir / "lexicon.json.gz", "wt") as handle:
        json.dump(rows, handle)
    with gzip.open(lexicon_dir / "spoken.jsonl.gz", "wt") as handle:
        handle.write(
            json.dumps({"term": "same", "voices": {"v1": "same", "v2": "same"}}) + "\n"
        )
        handle.write(
            json.dumps(
                {"term": "phonetic", "voices": {"v1": "fonetic", "v2": "fonetic"}}
            )
            + "\n"
        )
    monkeypatch.setattr("benchmark.LEXICON_DIR", lexicon_dir)
    out = tmp_path / "report.json"

    assert main(["--out", str(out), "--limit", "1"]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["terms"] == 1
    assert report["summary"]["by_diff_class"]["phonetic"] == 1


def test_stratified_slice_balances_classes_and_origins() -> None:
    rows = []
    classes = ("phonetic", "format", "case", "same")
    risks = {"phonetic": 1.0, "format": 0.6, "case": 0.2, "same": 0.0}
    for index in range(12):
        rows.append(
            {
                "term": f"term-{index}",
                "cats": ["gpu" if index % 2 else "toolbelt"],
                "diff_class": classes[index % 4],
                "risk": risks[classes[index % 4]],
                "origin": "local-inventory" if index % 2 else "shipped",
                "priority": 0.5,
                "voice_count": 2,
                "spoken_forms": [],
            }
        )
    selected = stratified_slice(rows, 12)
    assert len(selected) == 12
    assert len({row["term"] for row in selected}) == 12
    assert {row["diff_class"] for row in selected} == set(classes)
    assert {"shipped", "local-inventory"}.issubset({row["origin"] for row in selected})
    assert collections.Counter(row["origin"] for row in selected) == {
        "shipped": 6,
        "local-inventory": 6,
    }


def test_report_merges_local_inventory_and_writes_balanced_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexicon_dir = tmp_path / "shipped"
    lexicon_dir.mkdir()
    with gzip.open(lexicon_dir / "lexicon.json.gz", "wt") as handle:
        json.dump({"gpu": {"term": "GPU", "cats": ["gpu"]}}, handle)
    with gzip.open(lexicon_dir / "spoken.jsonl.gz", "wt") as handle:
        handle.write(
            json.dumps({"term": "GPU", "voices": {"v1": "GPU", "v2": "GPU"}}) + "\n"
        )
    extra_lexicon = tmp_path / "local-lexicon.json"
    extra_lexicon.write_text(
        json.dumps(
            {
                "tool": {
                    "term": "tool",
                    "cats": ["toolbelt"],
                    "priority": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    extra_spoken = tmp_path / "local-spoken.jsonl"
    extra_spoken.write_text(
        json.dumps({"term": "tool", "voices": {"v1": "toll", "v2": "toll"}}) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "summary": {"terms": 1},
                "terms": [{"term": "secret-local-term"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("benchmark.LEXICON_DIR", lexicon_dir)
    out = tmp_path / "report.json"
    stratified = tmp_path / "stratified.jsonl"

    assert (
        main(
            [
                "--out",
                str(out),
                "--extra-lexicon",
                str(extra_lexicon),
                "--extra-spoken",
                str(extra_spoken),
                "--extra-manifest",
                str(manifest),
                "--emit-stratified",
                str(stratified),
                "--stratified-size",
                "2",
            ]
        )
        == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["by_origin"] == {"local-inventory": 1, "shipped": 1}
    assert report["local_inventory"]["summary"]["terms"] == 1
    assert "terms" not in report["local_inventory"]
    assert "secret-local-term" not in out.read_text(encoding="utf-8")
    assert '"term"' not in out.read_text(encoding="utf-8")
    assert len(stratified.read_text(encoding="utf-8").splitlines()) == 2


def test_missing_lexicon_directory_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("benchmark.LEXICON_DIR", tmp_path / "missing")
    with pytest.raises(OSError):
        main(["--out", str(tmp_path / "report.json")])


def test_category_counts_are_multi_label_and_totals_exceed_rows() -> None:
    """Categories are labels, not a partition; a shared term counts in each."""
    shared = {
        "term": "shared",
        "cats": ["crate", "pypi"],
        "diff_class": "phonetic",
        "risk": 1.0,
        "origin": "shipped",
        "priority": 0.5,
        "voice_count": 2,
        "spoken_forms": [],
    }
    rows = [
        shared,
        {
            "term": "solo",
            "cats": ["cask"],
            "diff_class": "same",
            "risk": 0.0,
            "origin": "shipped",
            "priority": 0.5,
            "voice_count": 2,
            "spoken_forms": [],
        },
    ]
    summary = summarize(rows)
    assert summary["terms"] == 2
    assert sum(item["terms"] for item in summary["categories"]) == 3
    assert summary["categories"][0]["terms"] == 1
    assert summary["multi_label_terms"] == 1


def test_limit_keeps_the_highest_risk_rows(tmp_path: Path, monkeypatch) -> None:
    lexicon_dir = tmp_path / "shipped"
    lexicon_dir.mkdir()
    lexicon = {}
    spoken_lines = []
    for index in range(10):
        term = f"alpha{index}"
        lexicon[term] = {"term": term, "cats": ["pypi"]}
        heard = term.upper() if index >= 5 else term
        spoken_lines.append(
            json.dumps({"term": term, "voices": {"v1": heard, "v2": term}})
        )
    with gzip.open(lexicon_dir / "lexicon.json.gz", "wt") as handle:
        json.dump(lexicon, handle)
    with gzip.open(lexicon_dir / "spoken.jsonl.gz", "wt") as handle:
        handle.write("\n".join(spoken_lines) + "\n")
    monkeypatch.setattr("benchmark.LEXICON_DIR", lexicon_dir)

    limited = tmp_path / "limited.json"
    assert main(["--out", str(limited), "--limit", "5"]) == 0
    summary = json.loads(limited.read_text(encoding="utf-8"))["summary"]
    assert summary["terms"] == 5
    # Case-only errors outrank exact controls, so ranked truncation keeps all five.
    assert summary["risky_share"] == 0.0
    assert summary["by_diff_class"]["case"] == 5
    assert summary["multi_label_terms"] == 0


def test_public_manifest_drops_unreviewed_fields() -> None:
    """A future collector field must not ride into a shareable report."""
    projected = public_manifest(
        {
            "schema_version": 1,
            "summary": {"terms": 1},
            "sources": {"toolbelt": {"kept": 1, "seen": 1}},
            "collection_failures": {"cli-extension": "exit 1"},
            "non_claims": ["local"],
            "terms": [{"term": "private-name"}],
            "rows": [{"term": "another-private-name"}],
        }
    )
    assert "terms" not in projected
    assert "rows" not in projected
    assert projected["collection_failures"] == {"cli-extension": "exit 1"}
    assert projected["summary"] == {"terms": 1}
