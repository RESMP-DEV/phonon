"""Best-effort generic agentic mix (~30% of trajectory count) from open SFT sets.

Writes sft/train_generic.jsonl in the same {"messages","tools"} shape, or
nothing if the source schema is not a plain conversation list. Never fails
the pipeline.
"""
import json
import random
import sys
from pathlib import Path

N_TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 140
out = Path(sys.argv[1]) / "train_generic.jsonl"

try:
    from datasets import load_dataset
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if i > 4000:
            break
        conv = r.get("messages") or r.get("conversations")
        if not isinstance(conv, list) or len(conv) < 3:
            continue
        msgs = []
        ok = True
        for m in conv:
            role = m.get("role") or {"human": "user", "gpt": "assistant", "system": "system",
                                     "tool": "tool"}.get(m.get("from"), None)
            content = m.get("content") if "content" in m else m.get("value")
            if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
                ok = False
                break
            entry = {"role": role, "content": content}
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            msgs.append(entry)
        if not ok or msgs[-1]["role"] != "assistant":
            continue
        tools = r.get("tools")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                tools = None
        rows.append({"messages": msgs, "tools": tools if isinstance(tools, list) else None,
                     "source": "openthoughts-agent-v1"})
    random.Random(0).shuffle(rows)
    rows = rows[:N_TARGET]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generic] wrote {len(rows)} samples to {out}", file=sys.stderr)
except Exception as e:  # noqa: BLE001
    print(f"[generic] skipped: {type(e).__name__}: {e}", file=sys.stderr)
