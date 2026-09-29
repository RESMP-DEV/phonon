"""Corrupt short utterances with v1 calibrated corruption, pack with v1 to match length hist."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# v1 corrupt does `from paths import REAL_RATES`; load v1 first so that name is v1.paths.
V1 = Path("/home/user/phonon/research/synth_v1")
sys.path.insert(0, str(V1))
from corrupt import _load_v0, corrupt_text  # noqa: E402
from normalize import chat_messages, normalize_target  # noqa: E402
from pollution import has_identifier_dump  # noqa: E402

import importlib.util as _ilu

_v2_spec = _ilu.spec_from_file_location("synth_v2_paths", Path(__file__).resolve().parent / "paths.py")
assert _v2_spec and _v2_spec.loader
_v2p = _ilu.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2p)
BIN_NAMES = _v2p.BIN_NAMES
BINS = _v2p.BINS
CLEAN_PATH = _v2p.CLEAN_PATH
DATA_ROOT = _v2p.DATA_ROOT
FILTER_STATS = _v2p.FILTER_STATS
KIND_MIX = _v2p.KIND_MIX
LEXICON_PATH = _v2p.LEXICON_PATH
LOG_DIR = _v2p.LOG_DIR
PAIRS_PATH = _v2p.PAIRS_PATH
REAL_WERS_BY_BIN = _v2p.REAL_WERS_BY_BIN
SYSTEM_PROMPT = _v2p.SYSTEM_PROMPT
V1_FILTER_STATS = _v2p.V1_FILTER_STATS
V1_PAIRS = _v2p.V1_PAIRS
V1_REAL_WERS = _v2p.V1_REAL_WERS

get_error_model = _load_v0("errors").get_error_model


def bin_name(n: int) -> str:
    for (a, b), name in zip(BINS, BIN_NAMES, strict=True):
        if a <= n <= b:
            return name
    return "0"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hist_counts(rows: list[dict]) -> dict[str, int]:
    c = Counter(bin_name(len((r.get("target") or "").split())) for r in rows)
    return {k: int(c.get(k, 0)) for k in BIN_NAMES}


def hist_fracs(counts: dict[str, int]) -> dict[str, float]:
    n = sum(counts.values()) or 1
    return {k: counts[k] / n for k in BIN_NAMES}


def load_real_fracs() -> dict[str, float]:
    plan = DATA_ROOT / "tmp" / "hist_plan.json"
    if plan.exists():
        blob = json.loads(plan.read_text())
        return {k: float(blob["real"]["fracs"][k]) for k in BIN_NAMES}
    return {
        "1-5": 0.0621,
        "6-10": 0.1133,
        "11-20": 0.1709,
        "21-40": 0.2200,
        "41-80": 0.2163,
        "81+": 0.2174,
    }


def load_wers_for_bin(name: str, fallback: list[float]) -> list[float]:
    if REAL_WERS_BY_BIN.exists():
        blob = json.loads(REAL_WERS_BY_BIN.read_text())
        xs = list((blob.get("bins") or {}).get(name, {}).get("wers") or [])
        if xs:
            return xs
    return fallback


def load_fallback_wers() -> list[float]:
    if V1_REAL_WERS.exists():
        blob = json.loads(V1_REAL_WERS.read_text())
        wers = [float(x) for x in blob.get("wers") or []]
        if wers:
            return wers
    return [0.0] * 38 + [0.05] * 12 + [0.084] * 12 + [0.21] * 8 + [0.35] * 2


def v1_hit_scale() -> float:
    if V1_FILTER_STATS.exists():
        blob = json.loads(V1_FILTER_STATS.read_text())
        try:
            return float(blob.get("hit_scale") or 1.27)
        except (TypeError, ValueError):
            return 1.27
    return 1.27


def keep_short_text(text: str) -> tuple[bool, str]:
    target = normalize_target(text)
    if not target:
        return False, "empty_target"
    n = len(target.split())
    if n < 2 or n > 25:
        return False, "length"
    if has_identifier_dump(target):
        return False, "identifier_dump"
    if any(ord(ch) > 127 for ch in target):
        return False, "non_ascii"
    return True, "kept"


def build_short_pairs(
    clean: list[dict],
    substitutions: dict,
    rng: random.Random,
    hit_scale: float,
    fallback_wers: list[float],
) -> tuple[list[dict], dict]:
    kept = []
    drop_reason = Counter()
    for row in clean:
        text = row.get("text") or ""
        ok, reason = keep_short_text(text)
        if not ok:
            drop_reason[reason] += 1
            continue
        target = normalize_target(text)
        n = len(target.split())
        b = bin_name(n)
        wers = load_wers_for_bin(b, fallback_wers)
        target_wer = rng.choice(wers)
        terms = list(row.get("terms") or [])
        raw, applied = corrupt_text(target, rng, terms, substitutions, target_wer, hit_scale=hit_scale)
        if not raw.strip():
            drop_reason["empty_raw"] += 1
            continue
        uid = row.get("id") or f"{len(kept):05d}"
        kept.append(
            {
                "id": f"synth_v2_em_{uid}",
                "source": "synth_error_model",
                "input": raw,
                "target": target,
                "messages": chat_messages(raw, target, SYSTEM_PROMPT),
                "route": "error_model",
                "terms": terms,
                "error_classes_applied": applied,
                "prompt_id": row.get("prompt_id"),
                "setting": row.get("setting") or row.get("kind"),
                "kind": row.get("kind"),
                "target_wer_sampled": target_wer,
                "term_filter": reason,
                "synth_version": "v2_short",
            }
        )
        drop_reason["kept"] += 1
    stats = {"in": len(clean), "kept": len(kept), "drop_reason": dict(drop_reason)}
    return kept, stats


def split_by_bin(rows: list[dict]) -> dict[str, list[dict]]:
    out = {k: [] for k in BIN_NAMES}
    for row in rows:
        out[bin_name(len((row.get("target") or "").split()))].append(row)
    return out


def take_with_kind_mix(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or not rows:
        return []
    by: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by[str(row.get("kind") or "question")].append(row)
    for kind in by:
        rng.shuffle(by[kind])
    out: list[dict] = []
    used = {kind: 0 for kind in by}
    for kind, frac in KIND_MIX:
        k_n = min(int(round(n * frac)), len(by.get(kind, [])))
        out.extend(by.get(kind, [])[:k_n])
        used[kind] = k_n
    if len(out) < n:
        rest = []
        for kind, pool in by.items():
            rest.extend(pool[used.get(kind, 0) :])
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])
    return out[:n]


def mae_fracs(fracs: dict[str, float], real: dict[str, float]) -> float:
    return sum(abs(fracs[k] - real[k]) for k in BIN_NAMES) / len(BIN_NAMES)


def pack_rows(
    shorts: list[dict],
    em: list[dict],
    tts: list[dict],
    real_frac: dict[str, float],
    rng: random.Random,
    short_min: int = 6000,
    short_max: int = 8000,
) -> tuple[list[dict], dict]:
    """Keep all TTS and all 81+ EM. Fill each bin up to real_frac * N.

    Shorts fill 1-5/6-10/11-20/21-40 first; leftover target in 21-40/41-80 comes from
    v1 error-model rows. 81+ and 21-40 cannot reach real mass (v1 has almost no 81+
    and generation produced almost no 21-25). Search N so 1-5/6-10/11-20/41-80 sit
    close, with shorts in [6000, 8000].
    """
    s_bin = split_by_bin(shorts)
    e_bin = split_by_bin(em)
    t_bin = split_by_bin(tts)
    for k in BIN_NAMES:
        rng.shuffle(s_bin[k])
        rng.shuffle(e_bin[k])

    short_bins = ("1-5", "6-10", "11-20", "21-40")
    best = None
    for n_plan in range(12000, 22001, 50):
        take_s = {k: 0 for k in BIN_NAMES}
        take_e = {k: 0 for k in BIN_NAMES}
        counts = {k: 0 for k in BIN_NAMES}
        for k in BIN_NAMES:
            target = int(round(real_frac[k] * n_plan))
            tts_n = len(t_bin[k])
            remain = max(0, target - tts_n)
            if k in short_bins:
                s_n = min(len(s_bin[k]), remain)
                take_s[k] = s_n
                remain -= s_n
            if k == "81+":
                take_e[k] = len(e_bin[k])
            else:
                take_e[k] = min(len(e_bin[k]), remain)
            counts[k] = take_s[k] + take_e[k] + tts_n
        n_s = sum(take_s[k] for k in short_bins)
        if n_s < short_min:
            need = short_min - n_s
            for k in ("6-10", "11-20", "1-5"):
                extra = min(need, len(s_bin[k]) - take_s[k])
                take_s[k] += extra
                counts[k] += extra
                need -= extra
                if need <= 0:
                    break
            n_s = sum(take_s[k] for k in short_bins)
        if n_s < short_min or n_s > short_max:
            continue
        n = sum(counts.values()) or 1
        fracs = {k: counts[k] / n for k in BIN_NAMES}
        short_mae = sum(abs(fracs[k] - real_frac[k]) for k in ("1-5", "6-10", "11-20")) / 3
        mae_4180 = abs(fracs["41-80"] - real_frac["41-80"])
        mae = mae_fracs(fracs, real_frac)
        score = (short_mae, mae_4180, abs(n_s - 7000) / 7000, mae)
        cand = {
            "take_s": {k: take_s[k] for k in short_bins},
            "k2140": take_e["21-40"],
            "k4180": take_e["41-80"],
            "k1120_em": take_e["11-20"],
            "counts": counts,
            "fracs": fracs,
            "n": n,
            "n_plan": n_plan,
            "n_short": n_s,
            "mae": mae,
            "mae_no81": sum(abs(fracs[k] - real_frac[k]) for k in BIN_NAMES if k != "81+") / 5,
            "short_mae": short_mae,
            "score": score,
        }
        if best is None or cand["score"] < best["score"]:
            best = cand

    if best is None:
        take_s = {k: min(len(s_bin[k]), 2000) for k in short_bins}
        best = {
            "take_s": take_s,
            "k2140": min(len(e_bin["21-40"]), 800),
            "k4180": min(len(e_bin["41-80"]), int(0.35 * max(len(e_bin["41-80"]), 1))),
            "k1120_em": 0,
            "counts": {},
            "fracs": {},
            "n": 0,
            "n_short": sum(take_s.values()),
            "mae": 1.0,
            "mae_no81": 1.0,
            "short_mae": 1.0,
        }

    chosen = []
    for k, n_take in best["take_s"].items():
        chosen.extend(take_with_kind_mix(s_bin[k], n_take, rng))
    chosen.extend(e_bin["21-40"][: best["k2140"]])
    chosen.extend(e_bin["41-80"][: best["k4180"]])
    chosen.extend(e_bin["81+"])
    if best.get("k1120_em"):
        chosen.extend(e_bin["11-20"][: best["k1120_em"]])
    chosen.extend(tts)
    # stable-ish order: shorts, em, tts already concatenated; drop dups
    seen: set[tuple[str, str]] = set()
    pairs = []
    for row in chosen:
        key = (row.get("input") or "", row.get("target") or "")
        if key in seen or not key[0] or not key[1]:
            continue
        seen.add(key)
        pairs.append(row)
    rng.shuffle(pairs)
    counts = hist_counts(pairs)
    fracs = hist_fracs(counts)
    stats = {
        "pack": {
            "take_s": best["take_s"],
            "k2140": best["k2140"],
            "k4180": best["k4180"],
            "k81": len(e_bin["81+"]),
            "n_tts": len(tts),
            "n_short": sum(1 for r in pairs if r.get("synth_version") == "v2_short"),
            "n": len(pairs),
            "planned_n": best.get("n"),
            "planned_mae": best.get("mae"),
            "planned_mae_no81": best.get("mae_no81"),
            "planned_short_mae": best.get("short_mae"),
        },
        "counts": counts,
        "fracs": fracs,
        "pct": {k: 100.0 * fracs[k] for k in BIN_NAMES},
        "real_fracs": real_frac,
        "real_pct": {k: 100.0 * real_frac[k] for k in BIN_NAMES},
        "delta_pp": {k: 100.0 * (fracs[k] - real_frac[k]) for k in BIN_NAMES},
        "kind_counts": dict(Counter(r.get("kind") for r in pairs if r.get("synth_version") == "v2_short")),
    }
    return pairs, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hit-scale", type=float, default=0.0)
    parser.add_argument("--short-min", type=int, default=6000)
    parser.add_argument("--short-max", type=int, default=8000)
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    hit_scale = args.hit_scale if args.hit_scale > 0 else v1_hit_scale()
    fallback_wers = load_fallback_wers()
    model = get_error_model(prefer_sibling=True)
    substitutions = model.get("substitutions") or {}
    print(f"error_model source={model.get('source')} hit_scale={hit_scale}", flush=True)

    clean = load_jsonl(CLEAN_PATH)
    rng = random.Random(args.seed)
    shorts, short_stats = build_short_pairs(clean, substitutions, rng, hit_scale, fallback_wers)
    print(f"short pairs kept={len(shorts)} in={short_stats['in']} drop={short_stats['drop_reason']}", flush=True)

    v1 = load_jsonl(V1_PAIRS)
    em = [r for r in v1 if r.get("route") == "error_model"]
    tts = [r for r in v1 if r.get("route") == "tts_asr"]
    for row in em + tts:
        row.setdefault("synth_version", "v1")
    print(f"v1 em={len(em)} tts={len(tts)}", flush=True)

    real_frac = load_real_fracs()
    pairs, pack_stats = pack_rows(shorts, em, tts, real_frac, rng, args.short_min, args.short_max)
    write_jsonl(PAIRS_PATH, pairs)

    kind_n = Counter(r.get("kind") for r in shorts)
    stats = {
        "short": short_stats,
        "kind_generated": dict(kind_n),
        "kind_mix_target": {k: v for k, v in KIND_MIX},
        "hit_scale": hit_scale,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "n_v2_short": sum(1 for r in pairs if r.get("synth_version") == "v2_short"),
        "n_v1": sum(1 for r in pairs if r.get("synth_version") == "v1"),
        "n_em": sum(1 for r in pairs if r.get("route") == "error_model"),
        "n_tts": sum(1 for r in pairs if r.get("route") == "tts_asr"),
        "pairs_path": str(PAIRS_PATH),
        **pack_stats,
    }
    FILTER_STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in ("n_pairs", "n_v2_short", "n_v1", "n_em", "n_tts", "pct", "delta_pp", "kind_counts")}, indent=2))
    print(f"wrote {PAIRS_PATH} and {FILTER_STATS}", flush=True)
    if stats["n_v2_short"] < args.short_min:
        print(f"WARNING shorts in output {stats['n_v2_short']} < {args.short_min}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
