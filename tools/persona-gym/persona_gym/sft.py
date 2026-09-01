"""Export graded rollouts as SFT samples.

Each line: {"messages": [...], "tools": [...], "persona": str, "roll": str}
in OpenAI chat format. Reasoning fields are dropped (the student does not
think). Harness nudges ("answer now") and the empty assistant turns they
caused are removed so the student learns to finalize after digging. An
assistant turn whose tool calls were never executed is removed too.

Aux samples teach spoken_forms recovery from the synthetic gold manglings:
{"messages": [system, user(term + log lines), assistant(json)]}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .rollout import SYSTEM_PROMPT, TOOLS

NUDGE_PREFIXES = ("Reply with your final answer now", "Stop exploring.")

AUX_SYSTEM = ("You map dictated text to a developer's vocabulary. Given a canonical "
              "term and lines from dictated chat logs, list the garbled spoken "
              "forms of that term that appear in the lines. Reply with JSON only: "
              '{"term": str, "spoken_forms": [str]}.')


def clean_messages(rows: list[dict]) -> list[dict] | None:
    out: list[dict] = []
    i = 0
    while i < len(rows):
        m = rows[i]
        role = m.get("role")
        if role == "user" and (m.get("content") or "").startswith(NUDGE_PREFIXES):
            i += 1
            continue
        if role == "assistant":
            calls = m.get("tool_calls") or []
            content = m.get("content") or ""
            if calls:
                # keep only if the next messages are its tool results
                n_res = 0
                j = i + 1
                while j < len(rows) and rows[j].get("role") == "tool":
                    n_res += 1
                    j += 1
                if n_res != len(calls):
                    i = j
                    continue
                out.append({"role": "assistant", "content": content.strip(),
                            "tool_calls": [{
                                "type": "function", "id": c.get("id", ""),
                                "function": {"name": c["function"]["name"],
                                             "arguments": _args(c["function"]["arguments"])},
                            } for c in calls]})
                for k in range(n_res):
                    t = rows[i + 1 + k]
                    out.append({"role": "tool", "tool_call_id": t.get("tool_call_id", ""),
                                "content": t.get("content") or ""})
                i = j
                continue
            if not content.strip():
                i += 1
                continue
            out.append({"role": "assistant", "content": content.strip()})
            i += 1
            continue
        if role in ("system", "user", "tool"):
            out.append({k: m[k] for k in ("role", "content", "tool_call_id") if k in m})
        i += 1
    # must end with a final assistant text answer holding a json fence
    if not out or out[-1]["role"] != "assistant" or "```json" not in out[-1]["content"]:
        return None
    return out


def _args(a):
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a)
    except (json.JSONDecodeError, TypeError):
        return {"command": str(a)}


def export(personas: Path, rollouts: Path, out: Path, strict: Path | None = None,
           min_recall: float = 0.9, min_prec: float = 0.9, min_mangle: float = 0.7) -> None:
    report = json.loads((rollouts / "report.json").read_text())
    keep = set()
    if strict and strict.exists():
        keep = {tuple(x) for x in json.loads(strict.read_text())}
    else:
        for p, rolls in report.items():
            for rn, x in rolls.items():
                if (x.get("valid_json") and x["planted_recall"] >= min_recall
                        and x["precision_proxy"] >= min_prec
                        and x["mangling_recall"] >= min_mangle):
                    keep.add((p, rn))
    out.mkdir(parents=True, exist_ok=True)
    n = dropped = 0
    with open(out / "train_trajectories.jsonl", "w") as f:
        for p, rn in sorted(keep):
            traj = rollouts / p / rn / "trajectory.jsonl"
            if not traj.exists():
                continue
            rows = [json.loads(l) for l in open(traj)]
            msgs = clean_messages(rows)
            if msgs is None:
                dropped += 1
                continue
            msgs[0] = {"role": "system", "content": SYSTEM_PROMPT}
            f.write(json.dumps({"messages": msgs, "tools": TOOLS, "persona": p, "roll": rn},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"[sft] trajectories: {n} written, {dropped} dropped (no clean final answer)",
          file=sys.stderr)

    # Aux spoken-forms samples from gold manglings + the log lines that carry them.
    train_personas = {p for p, _ in keep}
    na = 0
    with open(out / "train_aux.jsonl", "w") as f:
        for gold in sorted((personas / "meta").glob("*/gold.json")):
            pname = gold.parent.name
            if pname not in train_personas:
                continue
            g = json.loads(gold.read_text())
            logs = []
            for lf in (personas / pname / "logs").glob("*.txt"):
                logs += lf.read_text(errors="ignore").splitlines()
            by_term: dict[str, list[str]] = {}
            for m in g.get("manglings", []):
                if m.get("term") and m.get("mangled"):
                    by_term.setdefault(m["term"], []).append(m["mangled"])
            for term, forms in by_term.items():
                pat = re.compile("|".join(re.escape(x) for x in forms), re.I)
                hits = [l for l in logs if pat.search(l)][:8]
                if not hits:
                    continue
                msgs = [{"role": "system", "content": AUX_SYSTEM},
                        {"role": "user", "content": f"Term: {term}\nLines:\n" + "\n".join(hits)},
                        {"role": "assistant", "content": json.dumps(
                            {"term": term, "spoken_forms": forms}, ensure_ascii=False)}]
                f.write(json.dumps({"messages": msgs, "persona": pname}, ensure_ascii=False) + "\n")
                na += 1
    print(f"[sft] aux spoken-forms samples: {na}", file=sys.stderr)
