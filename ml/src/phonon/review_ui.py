from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from phonon.audio import local_audio_path_for_row
from phonon.schema import now_iso, read_parquet_rows

DEFAULT_REVIEW_TITLE = "Phonon Label Review"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe_asset_name(row_id: str, index: int, suffix: str = ".wav") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", row_id).strip("._-")
    if not stem:
        stem = f"row_{index:05d}"
    return f"{index:05d}_{stem[:120]}{suffix}"


def _coerce_teacher_transcripts(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for name, transcript in value.items():
        text = transcript.get("text") if isinstance(transcript, dict) else transcript
        text = str(text or "").strip()
        if text:
            out[str(name)] = text
    return dict(sorted(out.items()))


def _draft_text(row: dict[str, Any], teacher_transcripts: dict[str, str]) -> tuple[str, str]:
    for key, source in (
        ("consensus_text", "teacher_consensus"),
        ("insert_text", "insert_text"),
        ("verbatim_text", "verbatim_text"),
        ("normalized_text", "normalized_text"),
    ):
        text = str(row.get(key) or "").strip()
        if text:
            return text, source
    if teacher_transcripts:
        name, text = max(teacher_transcripts.items(), key=lambda item: len(item[1]))
        return text, f"teacher:{name}"
    return "", "empty"


def _dedupe_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w.+#/-]+", " ", text.lower())).strip()


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_key: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        teachers = _coerce_teacher_transcripts(row.get("teacher_transcripts"))
        draft, _ = _draft_text(row, teachers)
        key = _dedupe_text(draft)
        if not key:
            key = str(row.get("id") or "")
        if key in by_key:
            duplicate_count += 1
            kept = by_key[key]
            kept.setdefault("_duplicate_ids", []).append(str(row.get("id") or ""))
            kept["_duplicate_count"] = int(kept.get("_duplicate_count") or 1) + 1
            continue
        copy = dict(row)
        copy["_dedupe_key"] = key
        copy["_duplicate_count"] = 1
        copy["_duplicate_ids"] = []
        by_key[key] = copy
    return list(by_key.values()), duplicate_count


def _teacher_transcript_groups(teacher_transcripts: dict[str, str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for name, text in sorted(teacher_transcripts.items()):
        key = _dedupe_text(text)
        match: dict[str, Any] | None = None
        for group in groups:
            if group["dedupe_key"] == key:
                match = group
                break
            ratio = SequenceMatcher(None, key, str(group["dedupe_key"]), autojunk=False).ratio()
            if ratio >= 0.975:
                match = group
                break
        if match is None:
            groups.append(
                {
                    "names": [name],
                    "text": text,
                    "dedupe_key": key,
                    "count": 1,
                }
            )
        else:
            match["names"].append(name)
            match["count"] = int(match["count"]) + 1
    return groups


def _find_audio_path(
    dataset_root: Path,
    row: dict[str, Any],
    dataset_row: dict[str, Any] | None,
) -> Path | None:
    existing = str(row.get("audio_path") or "").strip()
    if existing:
        path = Path(existing)
        if path.exists():
            return path
    if dataset_row:
        merged = {**dataset_row, **row}
        path = local_audio_path_for_row(dataset_root, merged)
        if path:
            return path
    return None


def _dataset_rows_by_id(dataset_root: Path, dataset: str, split: str) -> dict[str, dict[str, Any]]:
    dataset_dir = dataset_root / "datasets" / dataset
    if not dataset_dir.exists():
        return {}
    return {
        str(row["id"]): row
        for row in read_parquet_rows(dataset_dir, split, include_audio=False)
        if row.get("id")
    }


def _json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _review_row(
    row: dict[str, Any],
    dataset_row: dict[str, Any] | None,
    index: int,
    dataset_root: Path,
    audio_dir: Path,
    copy_audio: bool,
) -> dict[str, Any]:
    row_id = str(row.get("id") or dataset_row.get("id") if dataset_row else row.get("id") or "")
    merged = {**(dataset_row or {}), **row}
    teachers = _coerce_teacher_transcripts(merged.get("teacher_transcripts"))
    teacher_groups = _teacher_transcript_groups(teachers)
    draft, draft_source = _draft_text(merged, teachers)
    audio_path = _find_audio_path(dataset_root, merged, dataset_row)
    audio_url = ""
    copied_audio_path = ""
    if audio_path and copy_audio:
        asset_name = _safe_asset_name(row_id, index, audio_path.suffix or ".wav")
        audio_dir.mkdir(parents=True, exist_ok=True)
        target = audio_dir / asset_name
        if not target.exists() or target.stat().st_size != audio_path.stat().st_size:
            shutil.copy2(audio_path, target)
        audio_url = f"audio/{asset_name}"
        copied_audio_path = str(target)
    elif audio_path:
        audio_url = str(audio_path)

    return {
        "index": index,
        "id": row_id,
        "split": merged.get("split") or "",
        "source_id": merged.get("source_id") or "",
        "source_url": merged.get("source_url") or "",
        "clip_start": merged.get("clip_start"),
        "clip_end": merged.get("clip_end"),
        "duration_seconds": merged.get("duration_seconds"),
        "sha256_audio": merged.get("sha256_audio") or "",
        "audio_url": audio_url,
        "audio_path": str(audio_path) if audio_path else "",
        "copied_audio_path": copied_audio_path,
        "label_status": merged.get("label_status") or "",
        "draft_text": draft,
        "draft_source": draft_source,
        "verbatim_text": merged.get("verbatim_text") or "",
        "insert_text": merged.get("insert_text") or "",
        "normalized_text": merged.get("normalized_text") or "",
        "teacher_transcripts": teachers,
        "teacher_transcript_groups": teacher_groups,
        "teacher_names": list(teachers),
        "teacher_count": len(teachers),
        "teacher_group_count": len(teacher_groups),
        "consensus_teacher": merged.get("consensus_teacher") or "",
        "consensus_confidence": merged.get("consensus_confidence"),
        "consensus_status": merged.get("consensus_status") or "",
        "review_score": merged.get("review_score"),
        "teacher_disagreement": merged.get("teacher_disagreement")
        or merged.get("mean_pairwise_wer"),
        "max_pairwise_wer": merged.get("max_pairwise_wer"),
        "technical_term_hits": merged.get("technical_term_hits") or 0,
        "command_path_package_hits": merged.get("command_path_package_hits") or 0,
        "technical_terms": merged.get("technical_terms") or [],
        "term_spans": merged.get("term_spans") or [],
        "domain_tags": merged.get("domain_tags") or [],
        "difficulty_tags": merged.get("difficulty_tags") or [],
        "duplicate_count": int(merged.get("_duplicate_count") or 1),
        "duplicate_ids": merged.get("_duplicate_ids") or [],
        "curator_note": merged.get("curator_note") or "",
    }


def render_review_ui(payload: dict[str, Any]) -> str:
    style = """
    :root {
      color-scheme: light;
      --bg: #f8f9fa;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #576269;
      --line: #d9e0e4;
      --accent: #0e5f67;
      --accent-soft: #d8eef0;
      --warn: #865000;
      --warn-soft: #fef3e0;
      --bad: #9c2f38;
      --bad-soft: #fce8ec;
      --extra: #7a4a07;
      --extra-soft: #ffe8bc;
      --miss: #8f2632;
      --miss-soft: #f8d7dc;
      --same-soft: #eef3f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, textarea, input { font: inherit; letter-spacing: 0; }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button.primary.active-verdict { box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); }
    button.warn { background: var(--warn-soft); border-color: #dfbd80; }
    button.warn.active-verdict { box-shadow: inset 0 2px 4px rgba(0,0,0,0.15); }
    button.bad { background: var(--bad-soft); border-color: #e2a7ad; }
    button.bad.active-verdict { box-shadow: inset 0 2px 4px rgba(0,0,0,0.15); }
    :focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .app {
      display: grid;
      grid-template-rows: 48px minmax(0, 1fr) 28px;
      min-height: 100vh;
    }
    /* --- Command bar --- */
    .cmdbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 16px;
      height: 48px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    .cmdbar-nav { display: flex; gap: 4px; align-items: center; }
    .cmdbar-nav button { padding: 4px 8px; font-size: 13px; }
    .cmdbar-counter {
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
      white-space: nowrap;
    }
    .cmdbar-id {
      font-size: 12px;
      font-weight: 700;
      color: var(--ink);
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .cmdbar-status {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .cmdbar-spacer { flex: 1; min-width: 8px; }
    .cmdbar-throughput {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .cmdbar label {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
      cursor: pointer;
      white-space: nowrap;
    }
    /* --- Progress bar --- */
    .progress {
      flex: 1;
      height: 4px;
      display: flex;
      border-radius: 2px;
      overflow: hidden;
      background: var(--line);
      min-width: 60px;
    }
    .progress-seg { height: 100%; }
    .progress-seg.seg-approved { background: #2f7d55; }
    .progress-seg.seg-needs_elliot { background: #c48126; }
    .progress-seg.seg-skip { background: #9c2f38; }
    .progress-seg.seg-pending { background: var(--line); }
    .progress-seg.seg-current { outline: 1px solid #fff; outline-offset: -1px; }
    /* --- Workspace two-column --- */
    .workspace {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      min-height: 0;
    }
    .editor-col {
      padding: 16px 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .diffs-col {
      border-left: 1px solid var(--line);
      padding: 16px 20px;
      overflow-y: auto;
    }
    .diffs-col-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    /* --- Audio & speed --- */
    audio { width: 100%; display: block; }
    .speed-row {
      display: flex;
      gap: 4px;
      align-items: center;
      margin-top: 6px;
    }
    .speed-row button {
      padding: 3px 8px;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .speed-row button.active {
      background: var(--accent-soft);
      border-color: var(--accent);
    }
    /* --- Canonical editor --- */
    textarea {
      width: 100%;
      min-height: 220px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font-size: 15px;
      line-height: 1.45;
      color: var(--ink);
      background: #fff;
    }
    #notes {
      min-height: 56px;
      font-size: 13px;
    }
    /* --- Collapsed metadata --- */
    .meta-details {
      font-size: 13px;
      color: var(--muted);
    }
    .meta-details summary {
      cursor: pointer;
      user-select: none;
    }
    .meta-details .kv {
      margin-top: 8px;
      display: grid;
      grid-template-columns: 100px minmax(0,1fr);
      gap: 4px 10px;
      font-size: 13px;
    }
    .meta-details .kv div:nth-child(odd) { color: var(--muted); }
    /* --- Term pills --- */
    .pills { display: flex; flex-wrap: wrap; gap: 6px; }
    .pill {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 12px;
      color: #39474f;
      background: #fafafa;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    /* --- Verdict row --- */
    .verdict-row {
      display: flex;
      gap: 8px;
      padding-top: 4px;
    }
    .verdict-row button { flex: 1; }
    /* --- Teacher diff cards --- */
    .teacher-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .teacher-card-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .teacher-card-head .teacher-name { font-weight: 700; font-size: 13px; }
    .teacher-card-head kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, monospace;
      background: #f6f7f8;
      color: var(--muted);
    }
    .teacher-diff {
      font-size: 13px;
      line-height: 1.7;
      color: #253139;
      overflow-wrap: anywhere;
    }
    .diff-word {
      display: inline;
      border-radius: 4px;
      padding: 1px 3px;
      margin: 0 1px 3px 0;
    }
    .diff-word.same { background: transparent; }
    .diff-word.extra { background: var(--extra-soft); color: var(--extra); }
    .diff-word.missing {
      background: var(--miss-soft);
      color: var(--miss);
      text-decoration: line-through;
      text-decoration-thickness: 1px;
    }
    .diff-stat { color: var(--muted); font-size: 12px; }
    .diff-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }
    /* --- Status dots --- */
    .status-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; background: #aeb9bf; }
    .status-approved { background: #2f7d55; }
    .status-needs_elliot { background: #c48126; }
    .status-skip { background: #9c2f38; }
    /* --- Clip-list drawer --- */
    .drawer {
      position: fixed;
      top: 48px;
      right: 0;
      bottom: 28px;
      width: 360px;
      background: #fff;
      border-left: 1px solid var(--line);
      box-shadow: -4px 0 12px rgba(0,0,0,0.08);
      z-index: 20;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      overflow: hidden;
    }
    .drawer[hidden] { display: none; }
    .drawer-header {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .drawer-header span { font-size: 13px; font-weight: 700; }
    .drawer-search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
    }
    .drawer-tabs {
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .drawer-tabs button { padding: 5px 8px; font-size: 12px; }
    .drawer-tabs button.active { background: var(--accent-soft); border-color: var(--accent); }
    .drawer-list { overflow-y: auto; }
    .row-button {
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      text-align: left;
      padding: 10px 12px;
      display: grid;
      gap: 5px;
    }
    .row-button.active { background: var(--accent-soft); }
    .row-id { font-weight: 700; font-size: 12px; overflow-wrap: anywhere; }
    .row-sub { color: var(--muted); font-size: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    a { color: var(--accent); text-decoration-thickness: 1px; }
    /* --- Toast --- */
    .toast {
      position: fixed;
      bottom: 44px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--ink);
      color: #fff;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      z-index: 30;
      opacity: 0;
      transition: opacity 0.2s;
      pointer-events: none;
    }
    .toast.visible { opacity: 1; }
    /* --- Shortcut hints bar --- */
    .hints {
      position: sticky;
      bottom: 0;
      z-index: 5;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 16px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
      border-top: 1px solid var(--line);
    }
    .hints[hidden] { display: none; }
    .hints kbd {
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 11px;
      background: #f0f2f4;
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 1px 4px;
    }
    .hints button {
      padding: 0 6px;
      font-size: 11px;
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
    }
    /* --- Reduced motion --- */
    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; }
    }
    /* --- Responsive --- */
    @media (max-width: 1079px) {
      .workspace { grid-template-columns: 1fr; }
      .diffs-col { border-left: none; border-top: 1px solid var(--line); }
      .drawer { width: 100%; }
    }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(str(payload["title"]))}</title>
<style>{style}</style>
</head>
<body>
<div class="app">
  <header class="cmdbar" role="banner">
    <div class="cmdbar-nav">
      <button id="prevBtn" aria-label="Previous clip">&#9664;</button>
      <button id="nextBtn" aria-label="Next clip">&#9654;</button>
    </div>
    <span class="cmdbar-counter" id="counter"></span>
    <span class="cmdbar-id" id="clipId"></span>
    <span class="cmdbar-status" id="verdictStatus"></span>
    <div class="progress" id="progressBar" role="progressbar" aria-label="Review progress" aria-valuenow="0" aria-valuemax="{payload["row_count"]}"></div>
    <span class="cmdbar-throughput" id="throughput"></span>
    <label><input type="checkbox" id="autoplay" checked> Auto-play</label>
    <button id="drawerToggle" aria-label="Open clip list">&#8984;L</button>
    <button id="exportBtn">Export</button>
  </header>
  <div class="workspace">
    <main class="editor-col" role="main">
      <audio id="audio" controls preload="auto"></audio>
      <div class="speed-row">
        <button class="speed-btn active" data-rate="1">1&times;</button>
        <button class="speed-btn" data-rate="1.5">1.5&times;</button>
        <button class="speed-btn" data-rate="2">2&times;</button>
      </div>
      <textarea id="canonical" aria-label="Canonical transcript"></textarea>
      <details class="meta-details" id="metaDetails">
        <summary id="metaSummary">Details</summary>
        <div class="kv" id="metadata"></div>
      </details>
      <textarea id="notes" placeholder="Notes" aria-label="Review notes"></textarea>
      <div class="pills" id="terms"></div>
      <div class="verdict-row">
        <button class="primary" id="approveBtn"><kbd>A</kbd> Approve</button>
        <button class="warn" id="uncertainBtn"><kbd>U</kbd> Uncertain</button>
        <button class="bad" id="skipBtn"><kbd>S</kbd> Skip</button>
      </div>
    </main>
    <aside class="diffs-col" role="complementary">
      <div class="diffs-col-header">
        <span id="teacherCount"></span>
      </div>
      <div class="diff-legend">
        <span><span class="diff-word missing">red</span> missing from teacher</span>
        <span><span class="diff-word extra">amber</span> extra in teacher</span>
      </div>
      <div id="teachers"></div>
    </aside>
  </div>
  <footer class="hints" id="hints" role="contentinfo">
    <span><kbd>&#9664;</kbd><kbd>&#9654;</kbd> navigate</span>
    <span><kbd>Space</kbd> play/pause</span>
    <span><kbd>Esc</kbd> blur</span>
    <span><kbd>A</kbd>/<kbd>U</kbd>/<kbd>S</kbd> verdict</span>
    <span><kbd>1</kbd>&ndash;<kbd>9</kbd> use teacher</span>
    <span><kbd>E</kbd> edit</span>
    <span><kbd>[</kbd><kbd>]</kbd> speed</span>
    <button id="dismissHints" aria-label="Dismiss keyboard shortcuts">&times;</button>
  </footer>
</div>
<div class="drawer" id="drawer" role="dialog" aria-modal="true" aria-label="Clip list" hidden>
  <div class="drawer-header">
    <span>Clips</span>
    <button id="drawerClose" aria-label="Close clip list">&times;</button>
  </div>
  <div style="padding: 8px 12px; border-bottom: 1px solid var(--line);">
    <input id="search" class="drawer-search" type="search" placeholder="Filter clips" aria-label="Filter clips">
    <div class="drawer-tabs" id="tabs" style="margin-top: 8px;"></div>
  </div>
  <div class="drawer-list" id="rowList"></div>
</div>
<div class="toast" id="toast" aria-live="polite"></div>
<script id="review-data" type="application/json">{_json_script(payload)}</script>
<script>
const payload = JSON.parse(document.getElementById('review-data').textContent);
const rows = payload.rows;
const storageKey = 'phonon-review:' + payload.review_id;
let edits = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
let filter = 'all';
let query = '';
let current = 0;
let drawerOpen = false;
let currentSpeed = 1;
let sessionStart = null;
let verdictCount = 0;
let userInteracted = false;

function defaultState(row) {{
  return {{ verdict: 'pending', canonical_text: row.draft_text || '', notes: '', updated_at: '' }};
}}
function stateFor(row) {{
  if (!edits[row.id]) edits[row.id] = defaultState(row);
  return edits[row.id];
}}
function saveStore() {{
  localStorage.setItem(storageKey, JSON.stringify(edits));
}}
function saveCurrent() {{
  const row = rows[current];
  if (!row) return;
  const state = stateFor(row);
  const canonical = document.getElementById('canonical');
  const notes = document.getElementById('notes');
  state.canonical_text = canonical.value;
  state.notes = notes.value;
  state.updated_at = new Date().toISOString();
  saveStore();
}}
function visibleRows() {{
  const q = query.trim().toLowerCase();
  return rows.map((row, index) => [row, index]).filter(([row]) => {{
    const state = stateFor(row);
    const statusOk = filter === 'all' || state.verdict === filter;
    if (!statusOk) return false;
    if (!q) return true;
    return [row.id, row.source_id, row.draft_text, (row.technical_terms || []).join(' ')]
      .join(' ').toLowerCase().includes(q);
  }});
}}
function fmt(value, digits = 2) {{
  if (value === null || value === undefined || value === '') return 'n/a';
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : String(value);
}}
function statusClass(verdict) {{
  return 'status-dot status-' + verdict;
}}
/* --- Toast --- */
let toastTimer = null;
function showToast(msg) {{
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 3000);
}}
/* --- Progress bar --- */
function renderProgress() {{
  const bar = document.getElementById('progressBar');
  bar.innerHTML = '';
  let done = 0;
  const pct = rows.length ? (100 / rows.length) : 0;
  rows.forEach((row, i) => {{
    const v = stateFor(row).verdict;
    if (v !== 'pending') done += 1;
    const seg = document.createElement('div');
    seg.className = 'progress-seg seg-' + v + (i === current ? ' seg-current' : '');
    seg.style.width = pct + '%';
    bar.appendChild(seg);
  }});
  bar.setAttribute('aria-valuenow', String(done));
}}
/* --- Throughput --- */
function updateThroughput() {{
  let done = 0;
  rows.forEach(row => {{ if (stateFor(row).verdict !== 'pending') done += 1; }});
  verdictCount = done;
  const el = document.getElementById('throughput');
  if (!sessionStart || done === 0) {{
    el.textContent = done + ' done';
    return;
  }}
  const mins = (Date.now() - sessionStart) / 60000;
  const rate = mins > 0.1 ? (done / mins).toFixed(1) : '—';
  el.textContent = done + ' done · ' + rate + '/min';
}}
setInterval(updateThroughput, 10000);
/* --- Speed control --- */
function setSpeed(rate) {{
  currentSpeed = rate;
  const audio = document.getElementById('audio');
  audio.playbackRate = rate;
  document.querySelectorAll('.speed-btn').forEach(btn => {{
    btn.classList.toggle('active', Number(btn.dataset.rate) === rate);
  }});
}}
document.querySelectorAll('.speed-btn').forEach(btn => {{
  btn.onclick = () => setSpeed(Number(btn.dataset.rate));
}});
/* --- Auto-play --- */
function tryAutoPlay() {{
  if (!userInteracted) return;
  const cb = document.getElementById('autoplay');
  if (!cb.checked) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const audio = document.getElementById('audio');
  audio.playbackRate = currentSpeed;
  audio.play().catch(() => {{}});
}}
/* --- Drawer --- */
function toggleDrawer(forceState) {{
  drawerOpen = forceState !== undefined ? forceState : !drawerOpen;
  const el = document.getElementById('drawer');
  el.hidden = !drawerOpen;
  if (drawerOpen) {{
    renderDrawer();
    const search = document.getElementById('search');
    search.focus();
  }}
}}
function renderDrawer() {{
  renderTabs();
  renderList();
}}
function renderTabs() {{
  const statuses = [
    ['all', 'All'],
    ['pending', 'Pending'],
    ['approved', 'Approved'],
    ['needs_elliot', 'Uncertain'],
    ['skip', 'Skip'],
  ];
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = '';
  for (const [key, label] of statuses) {{
    const count = key === 'all' ? rows.length : rows.filter(row => stateFor(row).verdict === key).length;
    const btn = document.createElement('button');
    btn.className = filter === key ? 'active' : '';
    btn.textContent = `${{label}} ${{count}}`;
    btn.onclick = () => {{ saveCurrent(); filter = key; renderDrawer(); }};
    tabs.appendChild(btn);
  }}
}}
function renderList() {{
  const list = document.getElementById('rowList');
  list.innerHTML = '';
  for (const [row, index] of visibleRows()) {{
    const state = stateFor(row);
    const btn = document.createElement('button');
    btn.className = 'row-button' + (index === current ? ' active' : '');
    btn.innerHTML = `
      <div class="row-id"><span class="${{statusClass(state.verdict)}}"></span> ${{row.id}}</div>
      <div class="row-sub">
        <span>score ${{fmt(row.review_score, 1)}}</span>
        <span>teachers ${{row.teacher_count || 0}}</span>
      </div>`;
    btn.onclick = () => {{
      saveCurrent();
      current = index;
      toggleDrawer(false);
      renderAll();
    }};
    list.appendChild(btn);
  }}
}}
/* --- Metadata link helper --- */
function metadataLink(row) {{
  return row.source_url ? `<a href="${{row.source_url}}" target="_blank" rel="noreferrer">${{row.source_id || row.source_url}}</a>` : (row.source_id || '');
}}
function escapeHtml(value) {{
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}}
/* --- Diff engine (unchanged) --- */
function diffTokens(text) {{
  return String(text || '').match(/\\S+/g) || [];
}}
function diffKey(token) {{
  return token.toLowerCase().replace(/^[^\\w.+#/-]+|[^\\w.+#/-]+$/g, '');
}}
function diffOps(referenceText, teacherText) {{
  const ref = diffTokens(referenceText);
  const hyp = diffTokens(teacherText);
  const n = ref.length;
  const m = hyp.length;
  const dp = Array.from({{ length: n + 1 }}, () => Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i -= 1) {{
    for (let j = m - 1; j >= 0; j -= 1) {{
      dp[i][j] = diffKey(ref[i]) === diffKey(hyp[j])
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }}
  }}
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {{
    if (diffKey(ref[i]) === diffKey(hyp[j])) {{
      ops.push(['same', hyp[j]]);
      i += 1;
      j += 1;
    }} else if (dp[i + 1][j] >= dp[i][j + 1]) {{
      ops.push(['missing', ref[i]]);
      i += 1;
    }} else {{
      ops.push(['extra', hyp[j]]);
      j += 1;
    }}
  }}
  while (i < n) {{
    ops.push(['missing', ref[i]]);
    i += 1;
  }}
  while (j < m) {{
    ops.push(['extra', hyp[j]]);
    j += 1;
  }}
  return ops;
}}
function renderDiffHtml(referenceText, teacherText) {{
  const ops = diffOps(referenceText, teacherText);
  const counts = ops.reduce((acc, [kind]) => {{
    acc[kind] = (acc[kind] || 0) + 1;
    return acc;
  }}, {{ same: 0, missing: 0, extra: 0 }});
  const html = ops.map(([kind, token]) => {{
    const title = kind === 'missing' ? 'missing from this teacher' : kind === 'extra' ? 'extra in this teacher' : '';
    return `<span class="diff-word ${{kind}}" title="${{title}}">${{escapeHtml(token)}}</span>`;
  }}).join(' ');
  return {{
    html,
    stats: `missing ${{counts.missing || 0}} · extra ${{counts.extra || 0}}`,
  }};
}}
function renderTeacherDiffs() {{
  const row = rows[current];
  const canonical = document.getElementById('canonical').value;
  for (const diffEl of document.querySelectorAll('[data-diff-index]')) {{
    const index = Number(diffEl.getAttribute('data-diff-index'));
    const group = (row.teacher_transcript_groups || [])[index] || {{}};
    const text = group.text || '';
    const rendered = renderDiffHtml(canonical, text);
    diffEl.innerHTML = rendered.html;
    const stat = document.querySelector(`[data-diff-stat-index="${{index}}"]`);
    if (stat) stat.textContent = rendered.stats;
  }}
}}
/* --- Render editor column --- */
function renderEditor() {{
  const row = rows[current];
  if (!row) return;
  const state = stateFor(row);
  document.getElementById('counter').textContent = `${{current + 1}}/${{rows.length}}`;
  document.getElementById('clipId').textContent = row.id;
  const statusEl = document.getElementById('verdictStatus');
  statusEl.textContent = state.verdict === 'pending' ? '' : state.verdict.replace('_', ' ');
  const audio = document.getElementById('audio');
  audio.src = row.audio_url || '';
  audio.playbackRate = currentSpeed;
  document.getElementById('canonical').value = state.canonical_text;
  document.getElementById('notes').value = state.notes;
  /* Metadata summary line */
  const dur = fmt(row.duration_seconds, 1);
  document.getElementById('metaSummary').textContent = `Details · ${{dur}}s · ${{row.draft_source || 'unknown'}}`;
  document.getElementById('metadata').innerHTML = `
    <div>source</div><div>${{metadataLink(row)}}</div>
    <div>clip</div><div>${{fmt(row.clip_start, 1)}}s to ${{fmt(row.clip_end, 1)}}s</div>
    <div>duration</div><div>${{dur}}s</div>
    <div>draft</div><div>${{row.draft_source || ''}}</div>
    <div>score</div><div>${{fmt(row.review_score, 2)}}</div>
    <div>disagreement</div><div>${{fmt(row.teacher_disagreement, 3)}}</div>
    <div>consensus</div><div>${{row.consensus_teacher || 'n/a'}} · ${{fmt(row.consensus_confidence, 3)}}</div>
    <div>commands</div><div>${{row.command_path_package_hits || 0}}</div>
    <div>duplicates</div><div>${{row.duplicate_count || 1}}${{(row.duplicate_ids || []).length ? ' hidden: ' + row.duplicate_ids.slice(0, 5).join(', ') : ''}}</div>
    <div>note</div><div>${{escapeHtml(row.curator_note || '')}}</div>
    <div>sha256</div><div style="overflow-wrap:anywhere">${{row.sha256_audio || ''}}</div>
  `;
  /* Terms */
  const terms = document.getElementById('terms');
  terms.innerHTML = '';
  for (const term of row.technical_terms || []) {{
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = term;
    terms.appendChild(pill);
  }}
  if (!(row.technical_terms || []).length) {{
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = 'no extracted terms';
    terms.appendChild(pill);
  }}
  /* Verdict button active states */
  document.getElementById('approveBtn').classList.toggle('active-verdict', state.verdict === 'approved');
  document.getElementById('uncertainBtn').classList.toggle('active-verdict', state.verdict === 'needs_elliot');
  document.getElementById('skipBtn').classList.toggle('active-verdict', state.verdict === 'skip');
}}
/* --- Render diffs column --- */
function renderDiffs() {{
  const row = rows[current];
  if (!row) return;
  const teachers = document.getElementById('teachers');
  teachers.innerHTML = '';
  const groups = row.teacher_transcript_groups || Object.entries(row.teacher_transcripts || {{}}).sort().map(([name, text]) => ({{ names: [name], text, count: 1 }}));
  document.getElementById('teacherCount').textContent = `${{row.teacher_count || groups.length}} teachers · ${{groups.length}} unique`;
  groups.forEach((group, groupIndex) => {{
    const text = group.text || '';
    const names = group.names || [];
    const card = document.createElement('div');
    card.className = 'teacher-card';
    card.innerHTML = `
      <div class="teacher-card-head">
        <span class="teacher-name">${{escapeHtml(names.join(' + '))}}${{group.count > 1 ? ' (' + group.count + ')' : ''}}</span>
        <span class="diff-stat" data-diff-stat-index="${{groupIndex}}"></span>
        <button data-group-index="${{groupIndex}}">Use</button>
        <kbd>${{groupIndex + 1}}</kbd>
      </div>
      <div class="teacher-diff" data-diff-index="${{groupIndex}}"></div>
    `;
    card.querySelector('button').onclick = () => {{
      document.getElementById('canonical').value = text;
      saveCurrent();
      renderTeacherDiffs();
    }};
    teachers.appendChild(card);
  }});
  renderTeacherDiffs();
}}
/* --- Render all --- */
function renderAll() {{
  renderProgress();
  renderEditor();
  renderDiffs();
  updateThroughput();
  if (drawerOpen) renderDrawer();
}}
/* --- Find next pending --- */
function nextPendingIndex() {{
  for (let i = current + 1; i < rows.length; i++) {{
    if (stateFor(rows[i]).verdict === 'pending') return i;
  }}
  for (let i = 0; i < current; i++) {{
    if (stateFor(rows[i]).verdict === 'pending') return i;
  }}
  return -1;
}}
/* --- Set verdict with auto-advance --- */
function setVerdict(verdict) {{
  const row = rows[current];
  const state = stateFor(row);
  state.verdict = verdict;
  if (!sessionStart) sessionStart = Date.now();
  saveCurrent();
  const next = nextPendingIndex();
  if (next >= 0) {{
    current = next;
  }} else {{
    showToast('All clips reviewed in current set');
  }}
  renderAll();
  tryAutoPlay();
}}
/* --- Export (schema unchanged) --- */
function exportJsonl() {{
  saveCurrent();
  const reviewedAt = new Date().toISOString();
  const lines = rows.map(row => {{
    const state = stateFor(row);
    return JSON.stringify({{
      id: row.id,
      split: row.split,
      source_id: row.source_id,
      source_url: row.source_url,
      clip_start: row.clip_start,
      clip_end: row.clip_end,
      duration_seconds: row.duration_seconds,
      sha256_audio: row.sha256_audio,
      label_status: state.verdict === 'approved' ? 'human_locked' : state.verdict,
      review_verdict: state.verdict,
      verbatim_text: state.canonical_text.trim(),
      insert_text: state.canonical_text.trim(),
      normalized_text: '',
      term_spans: row.term_spans || [],
      technical_terms: row.technical_terms || [],
      review_notes: state.notes,
      draft_source: row.draft_source,
      reviewed_at: reviewedAt,
      teacher_transcripts: row.teacher_transcripts,
      teacher_transcript_groups: row.teacher_transcript_groups,
    }});
  }}).join('\\n') + '\\n';
  const blob = new Blob([lines], {{ type: 'application/jsonl' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = payload.export_name;
  a.click();
  URL.revokeObjectURL(url);
}}
/* --- Wiring --- */
document.getElementById('prevBtn').onclick = () => {{ saveCurrent(); current = Math.max(0, current - 1); renderAll(); tryAutoPlay(); }};
document.getElementById('nextBtn').onclick = () => {{ saveCurrent(); current = Math.min(rows.length - 1, current + 1); renderAll(); tryAutoPlay(); }};
document.getElementById('approveBtn').onclick = () => setVerdict('approved');
document.getElementById('uncertainBtn').onclick = () => setVerdict('needs_elliot');
document.getElementById('skipBtn').onclick = () => setVerdict('skip');
document.getElementById('exportBtn').onclick = exportJsonl;
document.getElementById('canonical').oninput = () => {{ saveCurrent(); renderTeacherDiffs(); }};
document.getElementById('notes').oninput = saveCurrent;
document.getElementById('drawerToggle').onclick = () => toggleDrawer();
document.getElementById('drawerClose').onclick = () => toggleDrawer(false);
document.getElementById('search').oninput = event => {{
  query = event.target.value;
  renderDrawer();
}};
/* Dismiss hints */
const hintsKey = storageKey + ':hints-dismissed';
if (localStorage.getItem(hintsKey)) document.getElementById('hints').hidden = true;
document.getElementById('dismissHints').onclick = () => {{
  document.getElementById('hints').hidden = true;
  localStorage.setItem(hintsKey, '1');
}};
/* Mark user interaction for autoplay */
document.addEventListener('click', () => {{ userInteracted = true; }}, {{ once: true }});
document.addEventListener('keydown', () => {{ userInteracted = true; }}, {{ once: true }});
/* --- Keyboard handler --- */
document.addEventListener('keydown', event => {{
  const tag = event.target ? event.target.tagName : '';
  const inInput = ['TEXTAREA', 'INPUT'].includes(tag);
  const isMeta = event.ctrlKey || event.metaKey;
  /* Drawer toggle: Ctrl/Cmd+L */
  if (isMeta && event.key === 'l') {{
    event.preventDefault();
    toggleDrawer();
    return;
  }}
  /* Ctrl/Cmd+E: export */
  if (isMeta && event.key === 'e') {{
    event.preventDefault();
    exportJsonl();
    return;
  }}
  /* Shift+Arrow: seek audio ±2s (works in both modes) */
  if (event.shiftKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {{
    event.preventDefault();
    const audio = document.getElementById('audio');
    audio.currentTime = Math.max(0, audio.currentTime + (event.key === 'ArrowLeft' ? -2 : 2));
    return;
  }}
  /* Escape: close drawer or blur active element */
  if (event.key === 'Escape') {{
    if (drawerOpen) {{ toggleDrawer(false); return; }}
    if (document.activeElement) document.activeElement.blur();
    return;
  }}
  /* Everything below is command-mode only (not in input/textarea) */
  if (inInput) return;
  if (event.key === ' ') {{
    event.preventDefault();
    const audio = document.getElementById('audio');
    if (audio.paused) {{ audio.playbackRate = currentSpeed; audio.play().catch(() => {{}}); }}
    else audio.pause();
    return;
  }}
  if (event.key === 'ArrowLeft') {{ document.getElementById('prevBtn').click(); return; }}
  if (event.key === 'ArrowRight') {{ document.getElementById('nextBtn').click(); return; }}
  if (event.key === 'r' || event.key === 'R') {{
    const audio = document.getElementById('audio');
    audio.currentTime = 0;
    audio.playbackRate = currentSpeed;
    audio.play().catch(() => {{}});
    return;
  }}
  if (event.key === '[') {{
    const rates = [1, 1.5, 2];
    const idx = rates.indexOf(currentSpeed);
    setSpeed(rates[Math.max(0, idx - 1)]);
    return;
  }}
  if (event.key === ']') {{
    const rates = [1, 1.5, 2];
    const idx = rates.indexOf(currentSpeed);
    setSpeed(rates[(idx + 1) % rates.length]);
    return;
  }}
  if (event.key === 'a') {{ document.getElementById('approveBtn').click(); return; }}
  if (event.key === 'u') {{ document.getElementById('uncertainBtn').click(); return; }}
  if (event.key === 's') {{ document.getElementById('skipBtn').click(); return; }}
  if (event.key === 'e' || event.key === 'Enter') {{
    event.preventDefault();
    document.getElementById('canonical').focus();
    return;
  }}
  /* Digit keys 1-9: use teacher group */
  const digit = Number(event.key);
  if (digit >= 1 && digit <= 9) {{
    const row = rows[current];
    const groups = row.teacher_transcript_groups || [];
    const group = groups[digit - 1];
    if (group) {{
      document.getElementById('canonical').value = group.text || '';
      saveCurrent();
      renderTeacherDiffs();
    }}
    return;
  }}
}});
renderAll();
</script>
</body>
</html>
"""


def build_review_ui(
    queue_file: Path,
    out: Path,
    dataset_root: Path,
    dataset: str,
    split: str,
    limit: int | None = None,
    json_out: Path | None = None,
    copy_audio: bool = True,
    title: str = DEFAULT_REVIEW_TITLE,
    dedupe: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(queue_file)
    input_row_count = len(rows)
    resolved_queue = str(queue_file.resolve())
    rows.sort(
        key=lambda row: (
            float(row.get("review_score") or 0),
            float(row.get("teacher_disagreement") or row.get("mean_pairwise_wer") or 0),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    dedupe_removed = 0
    if dedupe:
        rows, dedupe_removed = _dedupe_rows(rows)
    if limit is not None:
        rows = rows[:limit]

    dataset_rows = _dataset_rows_by_id(dataset_root, dataset, split)
    audio_dir = out.parent / "audio"
    prepared = [
        _review_row(
            row=row,
            dataset_row=dataset_rows.get(str(row.get("id") or "")),
            index=index,
            dataset_root=dataset_root,
            audio_dir=audio_dir,
            copy_audio=copy_audio,
        )
        for index, row in enumerate(rows)
    ]
    missing_audio = [row["id"] for row in prepared if not row["audio_url"]]
    created_at = now_iso()
    # Regenerating the same queue must retain browser-local review decisions. If
    # an existing JSON payload points at the same queue and row IDs, reuse its
    # identity; otherwise derive a stable identity from the queue path.
    previous_review_id = None
    if json_out and json_out.is_file():
        try:
            previous_payload = json.loads(json_out.read_text())
            previous_rows = previous_payload.get("rows", [])
            same_rows = [str(row.get("id") or "") for row in previous_rows] == [
                str(row.get("id") or "") for row in prepared
            ]
            if previous_payload.get("queue_file") == str(queue_file) and same_rows:
                previous_review_id = previous_payload.get("review_id")
        except (OSError, ValueError, TypeError):
            previous_review_id = None
    review_id = previous_review_id or re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        f"queue-{hashlib.sha256(resolved_queue.encode()).hexdigest()[:20]}",
    )
    payload = {
        "title": title,
        "created_at": created_at,
        "review_id": review_id,
        "queue_file": str(queue_file),
        "dataset": dataset,
        "split": split,
        "input_row_count": input_row_count,
        "row_count": len(prepared),
        "dedupe_enabled": dedupe,
        "dedupe_removed": dedupe_removed,
        "missing_audio_count": len(missing_audio),
        "missing_audio_ids": missing_audio[:50],
        "export_name": f"{out.stem}_human_review.jsonl",
        "rows": prepared,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_review_ui(payload))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {
        "html": str(out),
        "json": str(json_out) if json_out else None,
        "queue_file": str(queue_file),
        "dataset": dataset,
        "split": split,
        "input_rows": input_row_count,
        "rows": len(prepared),
        "dedupe_enabled": dedupe,
        "dedupe_removed": dedupe_removed,
        "audio_dir": str(audio_dir) if copy_audio else None,
        "missing_audio_count": len(missing_audio),
    }
