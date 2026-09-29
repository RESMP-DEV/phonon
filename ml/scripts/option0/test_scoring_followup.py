"""Checks for term boundaries, annotation coverage, and complete report tables."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates  # noqa: E402
import score_gates as scorer  # noqa: E402


@pytest.mark.parametrize("term,hyp,missed", [
    ("CUDA", "Use CUDA, please.", 0),
    ("CUDA", "Use cuda", 1),
    ("CUDA", "CUDAs", 1),
    ("API", "APIs", 1),
    ("torch", "torch.compile", 1),
    ("torch.compile", "Use torch.compile.", 0),
    ("Hugging Face", "Hugging   Face", 0),
    ("Hugging Face", "Hugging the Face", 1),
    ("C++", "C", 1),
    ("const float*", "const float", 1),
    ("pytorch-cuda=11.8", "pytorch-cuda 11.8", 1),
    ("torch.cuda.is_available()", "torch.cuda.is_available", 1),
    ("torch.cuda.is_available()", "Run torch.cuda.is_available().", 0),
])
def test_strict_terms_require_case_and_complete_token_sequence(term, hyp, missed):
    result = scorer.strict_technical_term_error_rate([[term]], [hyp])
    assert result["strict_technical_terms"] == 1
    assert result["strict_technical_term_misses"] == missed
    assert result["strict_technical_term_error_rate"] == missed


def test_lenient_metric_is_preserved_and_empty_annotations_are_unmeasured():
    result = scorer.term_metrics([["CUDA", "API", "CUDA"]], ["cuda APIs"])
    assert result["technical_terms"] == 3
    assert result["technical_term_error_rate"] == 0
    assert result["strict_technical_terms"] == 3
    assert result["strict_technical_term_error_rate"] == 1
    empty = scorer.term_metrics([[]], ["ordinary dictation"])
    assert empty["technical_terms"] == 0
    assert empty["technical_term_error_rate"] is None
    assert empty["strict_technical_term_error_rate"] is None


def test_actual_hard77_annotations_reach_both_scores(tmp_path):
    rows = gates.load_gate("hard77")
    assert len(rows) == 77
    assert sum(len(row["technical_terms"]) for row in rows) == 416
    predictions = tmp_path / "hard77.jsonl"
    predictions.write_text("".join(json.dumps({
        "id": row["id"], "hypothesis": " ; ".join(row["technical_terms"]),
    }) + "\n" for row in rows))
    meta = {"gate": "hard77", "clips": 77, "complete": True,
            "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()}
    result, diffs = scorer.score_prediction("hard77", rows, predictions, meta)
    assert result["complete"]
    assert result["technical_terms"] == result["strict_technical_terms"] == 416
    assert result["technical_term_error_rate"] == 0
    assert result["strict_technical_term_error_rate"] == 0
    assert len(diffs) == 77
    assert all(diff["strict_technical_term_error_rate"] in (None, 0) for diff in diffs)


def test_complete_report_has_personal_rows_and_both_term_metrics():
    models = []
    for tag, model_id, nominal in scorer.CANDIDATES:
        record = scorer.model_record(tag, model_id, nominal)
        for gate in gates.GATE_NAMES:
            annotated = gate not in scorer.PERSONAL_GATES or gate == "personal_cuda"
            record["gates"][gate] = {
                "complete": True, "status": "complete", "clip_count": 1,
                "fair_wer": 0.125, "strict_lowercase_wer": 0.25,
                "project_word_wer": 0.2,
                "technical_terms": 1 if annotated else 0,
                "technical_term_error_rate": 0.1 if annotated else None,
                "strict_technical_term_error_rate": 0.3 if annotated else None,
            }
        scorer.summarize_model(record)
        models.append(record)
    markdown = scorer.markdown({"models": models, "gate_availability": {}, "errors": []})
    assert "partial" not in markdown.lower()
    assert "hard77 strict term error" in markdown
    assert "wispr_holdout120 fair WER" in markdown
    assert "wispr_edit25 strict WER" in markdown
    assert "personal_cuda lenient term error" in markdown
    assert len(models) == 10
    for record in models:
        assert markdown.count("| " + record["tag"] + " |") == 3
        assert record["five_gate_mean_fair_wer"] == 0.125
