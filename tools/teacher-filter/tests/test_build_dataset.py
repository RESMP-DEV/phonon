from __future__ import annotations

import json
from pathlib import Path

from build_dataset import main, normalize, word_error_rate


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_three_teacher_tiering_and_pack_manifest(tmp_path: Path) -> None:
    slice_path = tmp_path / "slice.jsonl"
    slice_rows = [
        {"audio": f"audio-{index}.wav", "dur": 3.0 + index} for index in range(3)
    ]
    write_rows(slice_path, slice_rows)
    teacher_rows = {
        "cohere": [
            {"audio": "audio-0.wav", "hyp": "The RTX 6000 Pro"},
            {"audio": "audio-1.wav", "hyp": "one two three"},
            {"audio": "audio-2.wav", "hyp": "alpha"},
        ],
        "voxtral": [
            {"audio": "audio-0.wav", "hyp": "The RTX 6000 Pro"},
            {"audio": "audio-1.wav", "hyp": "one two three"},
            {"audio": "audio-2.wav", "hyp": "beta"},
        ],
        "qwen": [
            {"audio": "audio-0.wav", "hyp": "The RTX 6000 Pro."},
            {"audio": "audio-1.wav", "hyp": "one two four"},
            {"audio": "audio-2.wav", "hyp": "gamma delta epsilon"},
        ],
    }
    paths = {}
    for name, rows in teacher_rows.items():
        path = tmp_path / f"{name}.jsonl"
        write_rows(path, rows)
        paths[name] = path
    out = tmp_path / "structured"

    result = main(
        [
            "--cohere",
            str(paths["cohere"]),
            "--voxtral",
            str(paths["voxtral"]),
            "--qwen",
            str(paths["qwen"]),
            "--slice",
            str(slice_path),
            "--audio-root",
            str(tmp_path),
            "--out",
            str(out),
            "--auto-max-mean-wer",
            "0.25",
            "--derived-arm",
            "rnnoise-mp",
            "--derived-audio-root",
            str(tmp_path / "parallel"),
        ]
    )

    assert result == 0
    report = json.loads((out / "filter-report.json").read_text(encoding="utf-8"))
    assert report["tiers"] == {
        "auto": {"rows": 2, "seconds": 7.0, "percent_rows": 66.67},
        "review": {"rows": 0, "seconds": 0.0, "percent_rows": 0.0},
        "exclude": {"rows": 1, "seconds": 5.0, "percent_rows": 33.33},
    }
    pack = [
        json.loads(line)
        for line in (out / "pack-manifest.jsonl").read_text().splitlines()
    ]
    assert len(pack) == 2
    assert pack[0]["corrected"] == "The RTX 6000 Pro"
    assert pack[1]["label_source"] == "majority:cohere"
    derived = [
        json.loads(line)
        for line in (out / "derived-augmentation-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(derived) == 2
    assert derived[0]["audio"] == "rnnoise-mp/audio/rnnoise-mp-0.wav"
    assert derived[0]["arm"] == "rnnoise-mp"


def test_mismatched_audio_key_is_rejected(tmp_path: Path) -> None:
    slice_path = tmp_path / "slice.jsonl"
    write_rows(slice_path, [{"audio": "one.wav", "dur": 1.0}])
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    write_rows(one, [{"audio": "one.wav", "hyp": "text"}])
    write_rows(two, [{"audio": "two.wav", "hyp": "text"}])

    try:
        main(
            [
                "--cohere",
                str(one),
                "--voxtral",
                str(two),
                "--qwen",
                str(one),
                "--slice",
                str(slice_path),
                "--audio-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )
    except ValueError as error:
        assert "audio key mismatch" in str(error)
    else:
        raise AssertionError("expected mismatched audio keys to fail")


def test_wer_normalization_and_empty_behavior() -> None:
    assert normalize("  Hello, world's! ") == "hello world's"
    assert word_error_rate("same text", "Same, text!") == 0.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("one", "") == 1.0
