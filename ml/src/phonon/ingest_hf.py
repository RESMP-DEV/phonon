from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def load_install_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def download_hf_dataset(
    repo: str,
    intake_root: Path,
    allow_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    local_dir = intake_root / "hf" / repo.replace("/", "__")
    result = {
        "repo": repo,
        "local_dir": str(local_dir),
        "allow_patterns": allow_patterns or [],
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    result["snapshot_path"] = snapshot_path
    return result


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0]
        return int(out)
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def install_curated_config(config_path: Path, dataset_root: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_install_config(config_path)
    intake_root = dataset_root / "intake"
    log_dir = dataset_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / f"{config['name']}_install.progress.jsonl"
    results = []
    for item in config["datasets"]:
        result = download_hf_dataset(
            item["repo"],
            intake_root,
            allow_patterns=item.get("allow_patterns"),
            dry_run=dry_run,
        )
        result["local_dir_bytes"] = du_bytes(Path(result["local_dir"]))
        result["dataset_root_bytes"] = du_bytes(dataset_root)
        results.append(result)
        with progress_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    out = {
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "dry_run": dry_run,
        "results": results,
    }
    (log_dir / f"{config['name']}_install.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
