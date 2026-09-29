from __future__ import annotations

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
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #64727b;
      --line: #d9e0e4;
      --accent: #126b74;
      --accent-soft: #d8eef0;
      --warn: #9a5b11;
      --warn-soft: #f4e4c8;
      --bad: #9c2f38;
      --bad-soft: #f2d8dc;
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
    button.warn { background: var(--warn-soft); border-color: #dfbd80; }
    button.bad { background: var(--bad-soft); border-color: #e2a7ad; }
    .app {
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #fff;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-width: 0;
    }
    .brand { padding: 16px; border-bottom: 1px solid var(--line); }
    .brand h1 { font-size: 18px; line-height: 1.2; margin: 0 0 6px; }
    .brand .meta { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .filters { padding: 10px 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
    .search { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }
    .status-tabs { display: flex; flex-wrap: wrap; gap: 6px; }
    .status-tabs button { padding: 5px 8px; font-size: 12px; }
    .status-tabs button.active { background: var(--accent-soft); border-color: var(--accent); }
    .row-list { overflow: auto; }
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
    .status-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; background: #aeb9bf; }
    .status-approved { background: #2f7d55; }
    .status-needs_elliot { background: #c48126; }
    .status-skip { background: #9c2f38; }
    main { min-width: 0; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(246,247,248,0.96);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }
    .topbar-left, .topbar-right { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; min-width: 0; }
    .counter { color: var(--muted); font-size: 13px; }
    .content {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
      gap: 16px;
      padding: 16px 18px 28px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .panel-header {
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .panel-body { padding: 12px; }
    audio { width: 100%; display: block; }
    .clip-title { font-weight: 800; overflow-wrap: anywhere; }
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
    textarea {
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      line-height: 1.45;
      color: var(--ink);
      background: #fff;
    }
    #notes { min-height: 84px; }
    .teacher {
      border-top: 1px solid var(--line);
      padding: 10px 0;
      display: grid;
      gap: 6px;
    }
    .teacher:first-child { border-top: 0; padding-top: 0; }
    .teacher-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
    .teacher-name { font-weight: 700; font-size: 13px; }
    .teacher-text { font-size: 13px; line-height: 1.45; color: #253139; white-space: pre-wrap; overflow-wrap: anywhere; }
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
    .diff-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .diff-stat { color: var(--muted); font-size: 12px; }
    details.raw { font-size: 12px; color: var(--muted); }
    details.raw summary { cursor: pointer; }
    details.raw .teacher-text { margin-top: 6px; color: #253139; }
    .kv { display: grid; grid-template-columns: 120px minmax(0,1fr); gap: 6px 10px; font-size: 13px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    a { color: var(--accent); text-decoration-thickness: 1px; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }
      .content { grid-template-columns: 1fr; }
      .topbar { position: static; }
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
  <aside class="sidebar">
    <div class="brand">
      <h1>{html.escape(str(payload["title"]))}</h1>
      <div class="meta">{payload["row_count"]} clips · {payload["dedupe_removed"]} duplicates hidden · generated {html.escape(str(payload["created_at"]))}</div>
    </div>
    <div class="filters">
      <input id="search" class="search" type="search" placeholder="Filter clips">
      <div class="status-tabs" id="tabs"></div>
    </div>
    <div class="row-list" id="rowList"></div>
  </aside>
  <main>
    <div class="topbar">
      <div class="topbar-left">
        <button id="prevBtn">Prev</button>
        <button id="replayBtn">Replay</button>
        <button id="nextBtn">Next</button>
        <span class="counter" id="counter"></span>
      </div>
      <div class="topbar-right">
        <button class="primary" id="approveBtn">Approve</button>
        <button class="warn" id="uncertainBtn">Uncertain</button>
        <button class="bad" id="skipBtn">Skip</button>
        <button id="exportBtn">Export JSONL</button>
      </div>
    </div>
    <div class="content">
      <section class="panel">
        <div class="panel-header">
          <span class="clip-title" id="clipTitle"></span>
          <span id="scoreLine"></span>
        </div>
        <div class="panel-body">
          <audio id="audio" controls preload="metadata"></audio>
          <div style="height: 12px"></div>
          <textarea id="canonical"></textarea>
          <div style="height: 10px"></div>
          <textarea id="notes" placeholder="Notes"></textarea>
          <div style="height: 12px"></div>
          <div class="pills" id="terms"></div>
        </div>
      </section>
      <aside class="panel">
        <div class="panel-header"><span>Clip</span><span id="statusText"></span></div>
        <div class="panel-body">
          <div class="kv" id="metadata"></div>
        </div>
      </aside>
      <section class="panel" style="grid-column: 1 / -1">
        <div class="panel-header"><span>Teacher Diffs</span><span id="teacherCount"></span></div>
        <div class="panel-body" id="teachers"></div>
      </section>
    </div>
  </main>
</div>
<script id="review-data" type="application/json">{_json_script(payload)}</script>
<script>
const payload = JSON.parse(document.getElementById('review-data').textContent);
const rows = payload.rows;
const storageKey = 'phonon-review:' + payload.review_id;
let edits = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
let filter = 'all';
let query = '';
let current = 0;

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
    btn.onclick = () => {{ saveCurrent(); filter = key; renderAll(false); }};
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
        <span>disagree ${{fmt(row.teacher_disagreement, 3)}}</span>
        <span>terms ${{row.technical_term_hits || 0}}</span>
        <span>dupes ${{row.duplicate_count || 1}}</span>
      </div>`;
    btn.onclick = () => {{ saveCurrent(); current = index; renderAll(); }};
    list.appendChild(btn);
  }}
}}
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
function renderMain() {{
  const row = rows[current];
  const state = stateFor(row);
  document.getElementById('counter').textContent = `${{current + 1}} / ${{rows.length}}`;
  document.getElementById('clipTitle').textContent = row.id;
  document.getElementById('scoreLine').textContent = `score ${{fmt(row.review_score, 2)}} · disagreement ${{fmt(row.teacher_disagreement, 3)}}`;
  document.getElementById('statusText').textContent = state.verdict.replace('_', ' ');
  const audio = document.getElementById('audio');
  audio.src = row.audio_url || '';
  document.getElementById('canonical').value = state.canonical_text;
  document.getElementById('notes').value = state.notes;
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
  document.getElementById('metadata').innerHTML = `
    <div>source</div><div>${{metadataLink(row)}}</div>
    <div>clip</div><div>${{fmt(row.clip_start, 1)}}s to ${{fmt(row.clip_end, 1)}}s</div>
    <div>duration</div><div>${{fmt(row.duration_seconds, 1)}}s</div>
    <div>draft</div><div>${{row.draft_source || ''}}</div>
    <div>consensus</div><div>${{row.consensus_teacher || 'n/a'}} · ${{fmt(row.consensus_confidence, 3)}}</div>
    <div>commands</div><div>${{row.command_path_package_hits || 0}}</div>
    <div>duplicates</div><div>${{row.duplicate_count || 1}}${{(row.duplicate_ids || []).length ? ' hidden: ' + row.duplicate_ids.slice(0, 5).join(', ') : ''}}</div>
    <div>note</div><div>${{escapeHtml(row.curator_note || '')}}</div>
    <div>sha256</div><div style="overflow-wrap:anywhere">${{row.sha256_audio || ''}}</div>
  `;
  const teachers = document.getElementById('teachers');
  teachers.innerHTML = '';
  const groups = row.teacher_transcript_groups || Object.entries(row.teacher_transcripts || {{}}).sort().map(([name, text]) => ({{ names: [name], text, count: 1 }}));
  document.getElementById('teacherCount').textContent = `${{row.teacher_count || groups.length}} teachers · ${{groups.length}} unique transcripts`;
  const legend = document.createElement('div');
  legend.className = 'diff-legend';
  legend.innerHTML = '<span><span class="diff-word missing">red</span> missing from teacher</span><span><span class="diff-word extra">amber</span> extra in teacher</span>';
  teachers.appendChild(legend);
  groups.forEach((group, groupIndex) => {{
    const text = group.text || '';
    const names = group.names || [];
    const item = document.createElement('div');
    item.className = 'teacher';
    item.innerHTML = `
      <div class="teacher-head">
        <span class="teacher-name">${{escapeHtml(names.join(' + '))}}${{group.count > 1 ? ' (' + group.count + ')' : ''}}</span>
        <span class="diff-stat" data-diff-stat-index="${{groupIndex}}"></span>
        <button data-group-index="${{groupIndex}}">Use</button>
      </div>
      <div class="teacher-diff" data-diff-index="${{groupIndex}}"></div>
      <details class="raw"><summary>raw transcript</summary><div class="teacher-text"></div></details>
    `;
    item.querySelector('.teacher-text').textContent = text;
    item.querySelector('button').onclick = () => {{
      document.getElementById('canonical').value = text;
      saveCurrent();
      renderTeacherDiffs();
    }};
    teachers.appendChild(item);
  }});
  renderTeacherDiffs();
}}
function renderAll(includeMain = true) {{
  renderTabs();
  renderList();
  if (includeMain) renderMain();
}}
function setVerdict(verdict) {{
  const row = rows[current];
  const state = stateFor(row);
  state.verdict = verdict;
  saveCurrent();
  renderAll();
}}
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
document.getElementById('search').oninput = event => {{
  saveCurrent();
  query = event.target.value;
  renderAll(false);
}};
document.getElementById('prevBtn').onclick = () => {{ saveCurrent(); current = Math.max(0, current - 1); renderAll(); }};
document.getElementById('nextBtn').onclick = () => {{ saveCurrent(); current = Math.min(rows.length - 1, current + 1); renderAll(); }};
document.getElementById('replayBtn').onclick = () => {{
  const audio = document.getElementById('audio');
  audio.currentTime = 0;
  audio.play();
}};
document.getElementById('approveBtn').onclick = () => setVerdict('approved');
document.getElementById('uncertainBtn').onclick = () => setVerdict('needs_elliot');
document.getElementById('skipBtn').onclick = () => setVerdict('skip');
document.getElementById('exportBtn').onclick = exportJsonl;
document.getElementById('canonical').oninput = () => {{ saveCurrent(); renderTeacherDiffs(); }};
document.getElementById('notes').oninput = saveCurrent;
document.addEventListener('keydown', event => {{
  if (event.target && ['TEXTAREA', 'INPUT'].includes(event.target.tagName)) return;
  if (event.key === 'ArrowLeft') document.getElementById('prevBtn').click();
  if (event.key === 'ArrowRight') document.getElementById('nextBtn').click();
  if (event.key === ' ') {{ event.preventDefault(); document.getElementById('replayBtn').click(); }}
  if (event.key === 'a') document.getElementById('approveBtn').click();
  if (event.key === 'u') document.getElementById('uncertainBtn').click();
  if (event.key === 's') document.getElementById('skipBtn').click();
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
    review_id = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{queue_file}:{limit or 'all'}:{created_at}")
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
