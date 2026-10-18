from __future__ import annotations

import hashlib
import json
import stat
import wave
from pathlib import Path

import pytest
from derive import RNNN_MODEL_SHAS, main


def write_wav(path: Path, sample_rate: int = 48000, frames: int = 480) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x01" * frames)


def write_fake_ffmpeg(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "source = args[args.index('-i') + 1]\n"
        "shutil.copyfile(source, args[-1])\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_fake_deep_filter(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "out = Path(args[args.index('-o') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "source = Path(args[-1])\n"
        "shutil.copyfile(source, out / source.name)\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_derive_preserves_raw_and_records_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "raw.wav"
    write_wav(source)
    models = tmp_path / "models"
    models.mkdir()
    model = models / "mp.rnnn"
    model.write_bytes(b"model")
    monkeypatch.setitem(RNNN_MODEL_SHAS, "mp.rnnn", digest(model))
    ffmpeg = tmp_path / "ffmpeg"
    deep_filter = tmp_path / "deep-filter"
    write_fake_ffmpeg(ffmpeg)
    write_fake_deep_filter(deep_filter)
    out = tmp_path / "derived"
    source_before = digest(source)

    result = main(
        [
            str(source),
            "--out",
            str(out),
            "--variant",
            "rnnoise-mp",
            "--variant",
            "dfn-pf-atten6",
            "--ffmpeg",
            str(ffmpeg),
            "--deep-filter",
            str(deep_filter),
            "--rnnoise-root",
            str(models),
        ]
    )

    assert result == 0
    assert digest(source) == source_before
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == source_before
    assert metadata["raw_is_canonical"] is True
    assert [item["variant"] for item in metadata["variants"]] == [
        "rnnoise-mp",
        "dfn-pf-atten6",
    ]
    rnnoise = metadata["variants"][0]
    deep = metadata["variants"][1]
    assert "arnndn=model=" in rnnoise["command"][rnnoise["command"].index("-af") + 1]
    assert deep["command"][deep["command"].index("-a") + 1] == "6"
    assert rnnoise["parameters"]["sample_rate"] == 48000
    assert deep["parameters"]["sample_rate"] == 48000
    assert capsys.readouterr().out.startswith("{\n")


def test_rejects_non_native_source(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    write_wav(source, sample_rate=16000)
    with pytest.raises(ValueError, match="expected native 48 kHz"):
        main([str(source), "--out", str(tmp_path / "out")])
