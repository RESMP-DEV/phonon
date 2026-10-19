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
        triplet_path,
        "triplet",
        [(0, {"anchor": "one", "positive": "uno", "negative": "different"})],
    )
    second = public_seeds.convert(
        "sets",
        "synthetic/source",
        "0" * 40,
        set_path,
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


def test_safe_output_rejects_paths_outside_seed_root(tmp_path: Path) -> None:
    inside = tmp_path / "nested" / "seed.jsonl"
    assert public_seeds.safe_output(inside, tmp_path) == inside.resolve()
    try:
        public_seeds.safe_output(Path("/tmp") / "outside.jsonl", tmp_path)
    except public_seeds.SeedError:
        pass
    else:
        raise AssertionError("outside output accepted")
