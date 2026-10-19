from __future__ import annotations

import json
from pathlib import Path

import pytest
import triplets


def triplet() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "public_text",
        "source": "synthetic",
        "source_revision": "0" * 40,
        "anchor": "alpha",
        "positive": "beta",
        "negative": "gamma",
    }


def test_valid_triplet_is_accepted() -> None:
    triplets.validate_triplet(triplet())


def test_missing_and_duplicate_fields_are_rejected() -> None:
    invalid = triplet()
    invalid.pop("source_revision")
    with pytest.raises(triplets.TripletError):
        triplets.validate_triplet(invalid)

    duplicate = triplet()
    duplicate["negative"] = duplicate["anchor"]
    with pytest.raises(triplets.TripletError):
        triplets.validate_triplet(duplicate)


def test_private_jsonl_is_user_only_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "private.jsonl"
    triplets.write_jsonl(output, [triplet()], private=True)

    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text(encoding="utf-8"))["anchor"] == "alpha"
    assert not list(tmp_path.glob("*.tmp"))
