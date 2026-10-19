from __future__ import annotations

import gzip
import json
from pathlib import Path

import public_seeds
import pyarrow as pa
import pyarrow.parquet as pq


def test_set_pair_and_triplet_sources_convert(tmp_path: Path) -> None:
    triplet_path = tmp_path / "triplet.parquet"
    pq.write_table(
        pa.table(
            {
                "anchor": ["one", "two"],
                "positive": ["uno", "dos"],
                "negative": ["different", "otro"],
            }
        ),
        triplet_path,
    )
    set_path = tmp_path / "set.jsonl.gz"
    with gzip.open(set_path, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"set": ["long one", "short one"]}) + "\n")
        output.write(json.dumps({"set": ["long two", "short two"]}) + "\n")

    first = public_seeds.convert(
        "triplets",
        "synthetic/source",
        "0" * 40,
        "synthetic/triplet.parquet",
        "triplet",
        [(0, {"anchor": "one", "positive": "uno", "negative": "different"})],
    )
    second = public_seeds.convert(
        "sets",
        "synthetic/source",
        "0" * 40,
        "synthetic/sets.jsonl.gz",
        "set_pair",
        [
            (0, {"set": ["long one", "short one"]}),
            (1, {"set": ["long two", "short two"]}),
        ],
    )

    assert len(first[0]) == 1
    assert len(second[0]) == 2
    assert second[0][0]["anchor"] == "long one"
    assert second[0][0]["negative"] == "short two"


def test_stsb_low_similarity_pairs_are_rejected() -> None:
    sampled = [
        (0, {"sentence1": "one", "sentence2": "uno", "score": 0.9}),
        (1, {"sentence1": "two", "sentence2": "dos", "score": 0.1}),
    ]

    records, rejected = public_seeds.convert(
        "stsb",
        "synthetic/source",
        "0" * 40,
        "synthetic/stsb.parquet",
        "stsb",
        sampled,
    )

    assert rejected == 1
    assert len(records) == 1
    assert records[0]["anchor"] == "one"


def test_cyclic_negative_does_not_reuse_the_first_candidate() -> None:
    sampled = [
        (0, {"question": "q0", "context": "c0", "title": "same"}),
        (1, {"question": "q1", "context": "c1", "title": "same"}),
        (2, {"question": "q2", "context": "c2", "title": "other"}),
        (3, {"question": "q3", "context": "c3", "title": "third"}),
    ]

    assert public_seeds.cyclic_different_candidate(sampled, 0, "title") == sampled[2][1]
    assert public_seeds.cyclic_different_candidate(sampled, 2, "title") == sampled[3][1]


def test_safe_output_rejects_paths_outside_seed_root(tmp_path: Path) -> None:
    inside = tmp_path / "nested" / "seed.jsonl"
    assert public_seeds.safe_output(inside, tmp_path) == inside.resolve()
    try:
        public_seeds.safe_output(Path("/tmp") / "outside.jsonl", tmp_path)
    except public_seeds.SeedError:
        pass
    else:
        raise AssertionError("outside output accepted")


def test_output_aliases_reject_collisions(tmp_path: Path) -> None:
    source = tmp_path / "all-nli/triplet/train-00000-of-00001.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned")
    output = tmp_path / "output.jsonl"

    try:
        public_seeds.validate_output_aliases(tmp_path, output, output)
    except public_seeds.SeedError:
        pass
    else:
        raise AssertionError("identical output paths accepted")

    try:
        public_seeds.validate_output_aliases(tmp_path, source, tmp_path / "report.json")
    except public_seeds.SeedError:
        pass
    else:
        raise AssertionError("pinned source overwrite accepted")
