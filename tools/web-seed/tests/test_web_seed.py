from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web_seed


def test_official_results_outrank_non_official(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "title": "RTX 6000 Pro Blackwell Guide",
                "url": "https://blog.example/rtx",
                "highlights": [],
            },
            {
                "title": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "url": "https://www.nvidia.com/en-us/products/rtx-pro-6000/",
                "highlights": ["canonical name"],
            },
        ]
    }
    monkeypatch.setattr(
        web_seed.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(payload), ""),
    )
    args = argparse.Namespace(limit=5, domain=[], exq="exq")
    record = web_seed.search("RTX 6000 Pro Blackwell", args)
    assert record["results"][0]["official"] is True
    assert (
        record["results"][0]["title"]
        == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    )


def test_payload_excludes_zcode() -> None:
    payload = {
        "providers": ["exa"],
        "excluded_providers": ["zcode"],
        "records": [],
    }
    assert "zcode" not in payload["providers"]
    assert payload["excluded_providers"] == ["zcode"]
