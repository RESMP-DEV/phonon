"""Recover exact missing gate audio from documented HF mirrors using index range reads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["HF_HOME"] = "/data/hf"
os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/hf/hub"

from gates import REPORT, ROOT, gate_summary, hash_matches, load_gate  # noqa: E402

DEFAULT_REPOS = (
    "EVA-UNIT-01/youtube_freecodecamp_real_dev_courses_v0",
    "EVA-UNIT-01/youtube_technical_v2_gpumode_seed_20260525",
    "EVA-UNIT-01/youtube_technical_v2_gpumode_wide_20260525a",
    "EVA-UNIT-01/youtube_technical_v2_gpumode_latest_20260525b",
    "EVA-UNIT-01/youtube_technical_consensus_v0",
    "EVA-UNIT-01/youtube_technical_consensus_extended_v0",
    "EVA-UNIT-01/youtube_dev_tutorials_real_audio_v0",
)


def main():
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", dest="repos")
    parser.add_argument("--gate", action="append", dest="gates")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("/data/phonon_datasets/option0_supplemental"))
    args = parser.parse_args()
    gates = args.gates or ["course91", "uncertain48"]
    repos = args.repos or DEFAULT_REPOS
    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    api = HfApi(token=token)
    report_path = REPORT / "supplemental_audio.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"shards": {}, "errors": []}
    REPORT.mkdir(parents=True, exist_ok=True)

    def save():
        report["gate_availability"] = {gate: gate_summary(gate) for gate in gates}
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def missing():
        return {row["id"]: row["sha256_audio"] for gate in gates for row in load_gate(gate)
                if not hash_matches(row["audio_path"], row.get("sha256_audio"))}

    def scan(repo, revision, sibling, wanted):
        remote = f"datasets/{repo}@{revision}/{sibling.rfilename}"
        fs = HfFileSystem(token=token)
        with fs.open(remote, "rb", block_size=65536, cache_type="readahead") as source:
            parquet = pq.ParquetFile(source)
            matched = []
            mismatched = []
            aliases = []
            wanted_by_hash = {}
            for row_id, digest in wanted.items():
                wanted_by_hash.setdefault(digest, []).append(row_id)
            for batch in parquet.iter_batches(columns=["id", "sha256_audio"], batch_size=2048):
                for row in batch.to_pylist():
                    for target_id in wanted_by_hash.get(row["sha256_audio"], []):
                        if target_id != row["id"]:
                            aliases.append({"target_id": target_id, "source_id": row["id"],
                                            "sha256_audio": row["sha256_audio"]})
                    expected = wanted.get(row["id"])
                    if expected is None:
                        continue
                    if row["sha256_audio"] == expected:
                        matched.append(row)
                    else:
                        mismatched.append({**row, "expected_sha256_audio": expected})
        return {"repo": repo, "revision": revision, "filename": sibling.rfilename,
                "size_bytes": sibling.size, "exact_matches": matched, "hash_mismatches": mismatched,
                "hash_alias_matches": aliases, "index_match_version": 2,
                "index_columns": ["id", "sha256_audio"], "scanned_target_ids": sorted(wanted)}

    def recover_aliases(shard, local):
        targets = {row["id"]: row for gate in gates for row in load_gate(gate)}
        aliases = shard.get("hash_alias_matches", [])
        parquet = pq.ParquetFile(local)
        restored = []
        for group in range(parquet.num_row_groups):
            index_rows = parquet.read_row_group(group, columns=["id", "sha256_audio"]).to_pylist()
            matches = [(index, alias) for index, row in enumerate(index_rows) for alias in aliases
                       if row["id"] == alias["source_id"] and row["sha256_audio"] == alias["sha256_audio"]]
            if not matches:
                continue
            audio = parquet.read_row_group(group, columns=["audio"]).column("audio")
            for index, alias in matches:
                target = Path(targets[alias["target_id"]]["audio_path"])
                if not target.resolve().is_relative_to(Path("/data/phonon_segments_root").resolve()):
                    raise ValueError(f"Alias audio target is outside /data storage: {target}")
                if hash_matches(target, alias["sha256_audio"]):
                    continue
                data = audio[index].as_py()["bytes"]
                if hashlib.sha256(data).hexdigest() != alias["sha256_audio"]:
                    raise ValueError(f"Alias embedded audio hash mismatch: {alias['source_id']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if not hash_matches(target, alias["sha256_audio"]):
                    raise ValueError(f"Alias written audio hash mismatch: {target}")
                restored.append({**alias, "target_path": str(target)})
                print(f"Recovered exact-byte alias {alias['source_id']} -> {alias['target_id']}", flush=True)
        shard["restored_aliases"] = restored

    for repo in repos:
        wanted = missing()
        if not wanted:
            break
        try:
            info = api.dataset_info(repo, files_metadata=True)
            siblings = [sibling for sibling in info.siblings if sibling.rfilename.endswith(".parquet")]
            print(f"{repo}@{info.sha}: {len(siblings)} shards, {len(wanted)} missing target IDs", flush=True)
            matched_shards = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {}
                for sibling in siblings:
                    key = f"{repo}@{info.sha}/{sibling.rfilename}"
                    cached = report["shards"].get(key)
                    if (cached and cached.get("index_match_version") == 2
                            and set(wanted) <= set(cached.get("scanned_target_ids", []))):
                        if (any(row["id"] in wanted for row in cached["exact_matches"])
                                or any(row["target_id"] in wanted for row in cached["hash_alias_matches"])):
                            matched_shards.append(cached)
                        continue
                    futures[pool.submit(scan, repo, info.sha, sibling, wanted)] = key
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result = future.result()
                        report["shards"][key] = result
                        if result["exact_matches"] or result["hash_alias_matches"]:
                            matched_shards.append(result)
                        print(f"{key}: {len(result['exact_matches'])} exact matches, "
                              f"{len(result['hash_alias_matches'])} exact-hash aliases, "
                              f"{len(result['hash_mismatches'])} hash mismatches", flush=True)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}".replace(token, "[REDACTED]")
                        report["errors"].append({"source": key, "error": error})
                        print(f"FAILED {key}: {error}", flush=True)
                    save()
            for shard in matched_shards:
                directory = args.out / repo.rsplit("/", 1)[-1]
                print(f"Downloading exact-match shard {repo}/{shard['filename']}", flush=True)
                local = hf_hub_download(repo, shard["filename"], repo_type="dataset",
                                        revision=shard["revision"], token=token, local_dir=directory)
                shard["local_path"] = str(local)
                subprocess.run(
                    ["uv", "run", "--no-sync", "python",
                     str(ROOT / "scripts/option0/materialize_gate_audio.py"),
                     "--parquet-dir", str(Path(local).parent)], check=True, cwd=ROOT,
                )
                if shard.get("hash_alias_matches"):
                    recover_aliases(shard, local)
                save()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".replace(token, "[REDACTED]")
            report["errors"].append({"source": repo, "error": error})
            print(f"FAILED {repo}: {error}", flush=True)
            save()
    # Preserve recovery provenance across reruns that now search only the remaining missing clips.
    all_targets = [row for gate in gates for row in load_gate(gate)]
    targets_by_hash = {}
    for row in all_targets:
        targets_by_hash.setdefault(row["sha256_audio"], []).append(row)
    local_shards = []
    for shard in report["shards"].values():
        local = args.out / shard["repo"].rsplit("/", 1)[-1] / shard["filename"]
        if not local.is_file():
            continue
        sources = []
        for batch in pq.ParquetFile(local).iter_batches(columns=["id", "sha256_audio"]):
            for source in batch.to_pylist():
                for target in targets_by_hash.get(source["sha256_audio"], []):
                    if hash_matches(target["audio_path"], target["sha256_audio"]):
                        sources.append({"source_id": source["id"], "target_id": target["id"],
                                        "sha256_audio": target["sha256_audio"],
                                        "target_path": target["audio_path"]})
        local_shards.append({"repo": shard["repo"], "revision": shard["revision"],
                             "filename": shard["filename"], "local_path": str(local),
                             "size_bytes": local.stat().st_size, "verified_target_sources": sources})
    report["local_shards"] = local_shards
    report["recovered_target_ids"] = sorted({source["target_id"] for shard in local_shards
                                           for source in shard["verified_target_sources"]})
    save()
    for gate, summary in report["gate_availability"].items():
        print(f"{gate}: found {summary['found']}/{summary['labeled_rows']}, "
              f"missing {summary['missing']}", flush=True)
    print(f"Provenance: {report_path}", flush=True)


if __name__ == "__main__":
    main()
