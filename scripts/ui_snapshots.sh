#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=$(date +%Y%m%dT%H%M%S)-$$
snapshot_root=${PHONON_UI_SNAPSHOT_ROOT:-"$repo_root/build/ui-snapshots"}
output=${PHONON_UI_SNAPSHOT_DIR:-"$snapshot_root/run-$run_id"}
support=${PHONON_UI_FIXTURE_DIR:-"$snapshot_root/fixture-$run_id"}
mkdir -p "$snapshot_root"
if [[ -e "$output" || -e "$support" ]]; then
	echo "ui snapshot output and fixture directories must not already exist" >&2
	exit 2
fi
if [[ "$support" == "$output" || "$support" == "$output"/* || "$output" == "$support"/* ]]; then
	echo "ui snapshot output and fixture directories must be disjoint" >&2
	exit 2
fi
mkdir -p "$output" "$support"

python3 - "$output" "$support" <<'PY'
import json
import pathlib
import sys
import time

output = pathlib.Path(sys.argv[1])
support = pathlib.Path(sys.argv[2])

(support / "settings.json").write_text(json.dumps({
    "schema_version": 2,
    "streaming": True,
    "local_history": True,
    "screen_context": False,
    "microphone_priority": ["Synthetic Desk Mic"],
    "instant_mic": True,
    "sound_feedback": False,
    "shortcut_mode": "fn",
    "privacy_choice_made": True,
    "history_retention_days": 0,
}, indent=2))

(support / "dictionary.json").write_text(json.dumps({
    "schema_version": 1,
    "updated_at_unix_ms": 1_760_000_000_000,
    "entries": [
        {
            "phrase": "synthetic one",
            "replacement": "Synthetic One",
            "spoken_forms": ["sin the tick one"],
            "source": "ui-fixture",
            "starred": True,
            "usage_count": 4,
        },
        {
            "phrase": "Blackwell",
            "replacement": None,
            "spoken_forms": ["black well"],
            "source": "ui-fixture",
            "starred": False,
            "usage_count": 2,
        },
    ],
}, indent=2))

now_ms = int(time.time() * 1000)
history = [
    ("today", 0, "Use Blackwell for the synthetic benchmark.", "Use Blackwell for the synthetic benchmark.", 42, 20_000),
    ("older", 4 * 86_400_000, "Run the local model now.", "Run the local model now.", 18, 11_000),
]
for index, (name, age_ms, raw, final, words, duration_ms) in enumerate(history):
    item = support / "Corpus" / name
    item.mkdir(parents=True)
    (item / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "id": name,
        "created_at_unix_ms": now_ms - age_ms,
        "source": "ui-fixture",
        "audio_file": "audio.wav",
        "microphone": "Synthetic Desk Mic",
        "audio_duration_ms": duration_ms,
        "speech_detected": True,
        "raw_transcript": raw,
        "final_transcript": final,
        "intended_transcript": final,
        "screen_context_terms": [],
        "dictionary_corrections": [{"count": 1}],
        "llm": {"latency_ms": 180 + index * 40, "ttft_ms": 80 + index * 20, "tokens_per_second": 70 - index * 5},
    }, indent=2))
PY

(cd "$repo_root/bar" && swift build --disable-sandbox --product PhononBar >/dev/null)
binary="$repo_root/bar/.build/debug/PhononBar"
[[ -x "$binary" ]]

run_snapshot() {
	local mode=$1
	local page=${2:-}
	local demo_text=${3:-}
	local target="$output/$mode${page:+-$page}.png"
	(
		cd "$repo_root/bar"
		export PHONON_UI_SUPPORT_DIR="$support"
		export PHONON_UI_DEMO="$mode"
		export PHONON_UI_SNAPSHOT="$target"
		export PHONON_UI_EXIT_AFTER_SNAPSHOT=1
		if [[ -n "$page" ]]; then
			export PHONON_MAIN_PAGE="$page"
		fi
		if [[ -n "$demo_text" ]]; then
			export PHONON_UI_DEMO_TEXT="$demo_text"
		fi
		"$binary" >/dev/null
	)
	[[ -s "$target" ]]
}

run_snapshot main Home
run_snapshot main History
run_snapshot main Dictionary
run_snapshot main Settings
run_snapshot startup
run_snapshot idle
run_snapshot processing
run_snapshot compact
run_snapshot listening "" "Synthetic live transcript for the compact dock."

python3 - "$output" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
snapshots = {}
for path in sorted(output.glob("*.png")):
    snapshots[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
expected = {
    "main-Home.png", "main-History.png", "main-Dictionary.png",
    "main-Settings.png", "startup.png", "idle.png", "processing.png",
    "compact.png", "listening.png",
}
if set(snapshots) != expected:
    raise SystemExit(f"snapshot set mismatch: {sorted(set(snapshots) ^ expected)}")
(output / "receipt.json").write_text(json.dumps({
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "fixture": "synthetic/non-personal",
    "snapshot_count": len(snapshots),
    "snapshots": snapshots,
    "non_claims": [
        "Deterministic AppKit/SwiftUI cache snapshots prove rendering and layout only.",
        "They do not exercise model loading, dictation, accessibility navigation, or latency.",
    ],
}, indent=2) + "\n")
print(output / "receipt.json")
PY
