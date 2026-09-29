from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, NamedTuple

from phonon.exact_form import apply_exact_form_corrections


ResolveMode = Literal["broad", "gated", "oracle", "scored"]

PUNCT_WRAP = "`*_.,;:()[]{}"
FLEXIBLE_SEPARATOR_PATTERN = (
    r"(?:[\s._/@=:+\-]+|[\u2047\ufffd?]+|"
    r"\b(?:dot|underscore|dash|hyphen|slash|equals|colon)\b)+"
)


@dataclass(frozen=True)
class VocabCandidate:
    canonical: str
    category: str
    variants: tuple[str, ...]
    source: str = "manual"


@dataclass(frozen=True)
class ResolverTrace:
    stage: str
    canonical: str
    variant: str
    category: str
    source: str
    reason: str
    score: float = 0.0
    score_parts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairResult:
    raw_text: str
    exact_text: str
    final_text: str
    exact_rules: tuple[str, ...]
    vocab_traces: tuple[ResolverTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "exact_text": self.exact_text,
            "final_text": self.final_text,
            "exact_rules": list(self.exact_rules),
            "vocab_traces": [trace.__dict__ for trace in self.vocab_traces],
        }


class _VariantEntry(NamedTuple):
    candidate: VocabCandidate
    variant: str


HARDCODED_CANDIDATES = [
    "nvcc --version",
    "nvidia-smi",
    "-arch=sm_90",
    "-lineinfo",
    "-Xptxas -v",
    "cudaMemcpyAsync",
    "cudaMemcpy2DAsync",
    "cudaMallocAsync",
    "cudaFreeAsync",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaStreamCreateWithFlags",
    "cudaStreamNonBlocking",
    "cudaEventRecord",
    "cudaFuncSetAttribute",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaHostRegister",
    "cudaMemPrefetchAsync",
    "cudaStreamWaitEvent",
    "blockIdx.x",
    "blockDim.x",
    "threadIdx.x",
    "warpSize",
    "nvcuda::wmma",
    "CUTLASS",
    "CuTe",
    "GEMM",
    "cuBLAS",
    "cuBLASLt",
    "cublasGemmEx",
    "FP16",
    "FP32",
    "cuDNN",
    "Winograd",
    "Triton",
    "triton.autotune",
    "enforce_eager=True",
    "llama.cpp",
    "CUDA_VISIBLE_DEVICES=0,1",
    "instance_group",
    "dynamic_batching",
    "@vercel/speed-insights",
    "Qwen",
]


MANUAL_VARIANTS = {
    "cudaMemcpyAsync": ("cuda mem copy async", "cuda memcpy async", "cuda me copy async", "cuda memcopy async"),
    "cudaMemcpy2DAsync": ("cuda mem copy 2d async", "cuda memocopy 2d async", "cuda memcpy 2d async"),
    "cudaMallocAsync": ("cuda malloc async", "cuda malak async", "cudamalik async"),
    "cudaFreeAsync": ("cuda free async",),
    "cudaDeviceSynchronize": ("cuda device synchronize", "code device synchronize"),
    "cudaStreamSynchronize": ("cuda stream synchronize",),
    "cudaStreamCreateWithFlags": ("cuda stream create with flags", "code stream create with flags"),
    "cudaStreamNonBlocking": ("cuda stream non blocking", "cuda stream non-blocking", "code stream non blocking"),
    "cudaEventRecord": ("cuda event record", "cuda record"),
    "cudaFuncSetAttribute": ("cuda func set attribute",),
    "cudaGetLastError": ("cuda get last error",),
    "cudaPeekAtLastError": ("cuda peak at last error", "cuda peek at last error"),
    "cudaHostRegister": ("cuda host register", "code host registers"),
    "cudaMemPrefetchAsync": ("cuda mem prefetch async", "cudam and prefetch async"),
    "cudaStreamWaitEvent": ("cuda stream wait event", "cuda stream weight event", "cuda's stream wait event"),
    "blockIdx.x": ("block idx.x", "blockidx.x", "block id x", "block id.x"),
    "blockDim.x": ("block dim.x", "blockdim.x", "block dim x", "block 10.x"),
    "threadIdx.x": ("thread idx.x", "threadidx.x", "threadsidx.x", "red idx.x"),
    "warpSize": ("warp size",),
    "nvcuda::wmma": ("nvcu.wmma", "nvcud.wmma", "nv cuda wmma", "nvcuda wmma"),
    "CUTLASS": ("cutlass",),
    "CuTe": ("cute", "q t", "qt"),
    "GEMM": ("gem",),
    "cuBLAS": ("kublas", "kublos", "kubloss", "kublus", "qblas", "qblos", "qbloss", "cublas"),
    "cuBLASLt": ("kublas lt", "kublast lt", "cublas lt", "cu blas l t"),
    "cublasGemmEx": ("kublas gem x", "cublas gem x", "cublas gem ex"),
    "cuDNN": ("qdnn", "kudianen", "cudnn"),
    "triton.autotune": ("triton.autotune", "triton dot autotune"),
    "enforce_eager=True": ("enforce eager equals true",),
    "llama.cpp": ("wama c", "lama cpp", "llama cpp"),
    "CUDA_VISIBLE_DEVICES=0,1": ("cuda visible devices equals 0,1", "cuda visible devices equals zero one"),
    "instance_group": ("instance underscore group",),
    "dynamic_batching": ("dynamic underscore batching",),
    "@vercel/speed-insights": ("versal slash speed insights", "vercel slash speed insights"),
    "nvcc --version": ("nvcc-version", "nvcc version", "nbcc-version"),
    "-Xptxas -v": ("xptxas v", "xptxas - v", "-xptxas - - v", "x ptxas v"),
    "-arch=sm_90": ("arch equals sm90", "arch equals sm 90", "arch equals sm90-1"),
    "-lineinfo": ("line info", "1 info"),
    "Qwen": ("quen", "qwen"),
}


AMBIGUOUS_CANONICALS = {"CUDA", "CU", "CuTe", "GEMM", "Qwen"}

CUDA_CONTEXT_MARKERS = {
    "cuda",
    "kernel",
    "block",
    "thread",
    "warp",
    "stream",
    "nsight",
    "nvcc",
    "nvidia",
    "cutlass",
    "triton",
    "winograd",
    "ptx",
    "sass",
    "sm90",
    "sm_90",
    "tensor core",
}

MODEL_CONTEXT_MARKERS = {
    "model",
    "tokenizer",
    "hugging face",
    "transformer",
    "llm",
    "moe",
    "inference",
    "vllm",
    "qwen",
    "quen",
    "llama",
}

QT_CONTEXT_MARKERS = {"qt", "qml", "widgets", "gui", "signal", "slot", "creator", "qmake", "cmake"}


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def index_key(text: str) -> str:
    match = re.search(r"[A-Za-z0-9]+", text.lower())
    return match.group(0) if match else normalize_key(text)


def text_index_keys(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("::", " ")
    text = text.replace("_", " ").replace(".", " dot ")
    text = text.replace("@", " at ").replace("/", " slash ").replace("-", " ")
    text = re.sub(r"=+", " equals ", text)
    return " ".join(text.split())


def category_for(term: str) -> str:
    if term in {"blockIdx.x", "blockDim.x", "threadIdx.x", "warpSize", "nvcuda::wmma"}:
        return "cuda_indexing_intrinsics"
    if term.startswith("cuda") or term.startswith("cu") or term in {"CUTLASS", "CuTe", "GEMM", "FP16", "FP32", "Winograd"}:
        return "cuda_api_libraries"
    if term.startswith("-") or "--" in term or "=" in term or term in {"nvcc --version", "nvidia-smi"}:
        return "commands_flags_env"
    if term.startswith("@") or term in {"instance_group", "dynamic_batching"}:
        return "serving_web_config"
    if term == "Qwen":
        return "near_neighbor_controls"
    return "other_technical"


def source_domain_for(term: str, item: dict[str, Any] | None = None) -> str:
    if item:
        item_type = str(item.get("type") or "").lower()
        contexts = " ".join(str(ctx) for ctx in item.get("contexts") or [])[:1000].lower()
        term_sources = " ".join(str(src) for src in item.get("sources") or [])[:1000].lower()
        blob = f"{item_type} {contexts} {term_sources}"
        if any(marker in blob for marker in ("cuda", ".cu", "cutlass", "triton", "nvidia", "kernel")):
            return "cuda_perf"
        if any(marker in blob for marker in ("next", "react", "vercel", "typescript", "tsx", "shadcn")):
            return "web_app"
        if any(marker in blob for marker in ("pyproject", "pytest", "uv ", "python", ".py")):
            return "python_packaging"
    if re.search(r"cuda|cublas|cudnn|cutlass|triton|nsight|nvcc|blockIdx|threadIdx|sm[_-]?\d+", term, re.I):
        return "cuda_perf"
    if re.search(r"qwen|llama|vllm|transformers|tokenizer|tensorRT|safetensors", term, re.I):
        return "ml_infra"
    if re.search(r"next|react|vercel|shadcn|typescript|tailwind|supabase", term, re.I):
        return "web_app"
    if re.search(r"pyproject|pytest|ipython|uv|python", term, re.I):
        return "python_packaging"
    return "general_technical"


def variants_for(term: str, spoken_variants: list[str] | None = None) -> tuple[str, ...]:
    variants = {term, split_identifier(term)}
    variants.update(MANUAL_VARIANTS.get(term, ()))
    if spoken_variants:
        variants.update(spoken_variants)
    variants = {variant.strip() for variant in variants if variant and len(normalize_key(variant)) >= 2}
    return tuple(sorted(variants, key=len, reverse=True))


def load_vocab_candidates(paths: Iterable[Path], extra_terms: Iterable[str] = ()) -> list[VocabCandidate]:
    by_term: dict[str, set[str]] = defaultdict(set)
    source_by_term: dict[str, str] = {}
    for term in [*HARDCODED_CANDIDATES, *extra_terms]:
        by_term[term].update(variants_for(term))
        source_by_term[term] = source_domain_for(term)
    for path in paths:
        data = json.loads(path.read_text())
        terms = data.get("terms", []) if isinstance(data, dict) else data
        for item in terms:
            if not isinstance(item, dict):
                continue
            term = str(item.get("canonical") or "")
            if not term or len(term) > 80:
                continue
            if int(item.get("difficulty") or 0) < 3 and term not in HARDCODED_CANDIDATES:
                continue
            by_term[term].update(variants_for(term, item.get("spoken_variants") or []))
            source_by_term[term] = source_domain_for(term, item)
    candidates = [
        VocabCandidate(term, category_for(term), tuple(sorted(variants, key=len, reverse=True)), source_by_term.get(term, "repo"))
        for term, variants in by_term.items()
        if term and variants
    ]
    return sorted(candidates, key=lambda c: (c.category, c.canonical.lower()))


def has_any_marker(text: str, markers: set[str]) -> bool:
    low = text.lower()
    return any(marker in low for marker in markers)


def is_symbolic_exact_form(term: str) -> bool:
    return bool(re.search(r"[._/@=-]|[a-z][A-Z]|[A-Z]{2,}", term))


def variant_is_risky(variant: str, canonical: str) -> bool:
    key = normalize_key(variant)
    if canonical in AMBIGUOUS_CANONICALS:
        return True
    if len(key) <= 3:
        return True
    return variant.lower() in {"cute", "gem", "cue", "cu", "qt", "quen"}


def candidate_allowed_by_context(candidate: VocabCandidate, text: str) -> bool:
    if candidate.canonical == "CuTe":
        return has_any_marker(text, {"cutlass", "gemm"})
    if candidate.canonical == "GEMM":
        return has_any_marker(text, CUDA_CONTEXT_MARKERS | {"matmul", "matrix", "blas"})
    if candidate.canonical == "Qwen":
        return has_any_marker(text, MODEL_CONTEXT_MARKERS)
    if candidate.canonical in {"CUDA", "CU"}:
        return has_any_marker(text, CUDA_CONTEXT_MARKERS)
    if candidate.source == "cuda_perf":
        return has_any_marker(text, CUDA_CONTEXT_MARKERS)
    if candidate.source == "ml_infra":
        return has_any_marker(text, MODEL_CONTEXT_MARKERS)
    return True


def context_score(candidate: VocabCandidate, text: str) -> tuple[float, list[str]]:
    score = 0.0
    parts: list[str] = []
    if candidate.canonical == "CuTe":
        if has_any_marker(text, {"cutlass", "gemm"}):
            score += 4.0
            parts.append("cuda_layout_context:+4.0")
        return score, parts
    if candidate.canonical == "GEMM":
        if has_any_marker(text, CUDA_CONTEXT_MARKERS | {"matmul", "matrix", "blas"}):
            score += 4.0
            parts.append("gemm_context:+4.0")
        return score, parts
    if candidate.canonical == "Qwen":
        if has_any_marker(text, MODEL_CONTEXT_MARKERS):
            score += 4.0
            parts.append("model_context:+4.0")
        return score, parts
    if candidate.canonical in {"CUDA", "CU"}:
        if has_any_marker(text, CUDA_CONTEXT_MARKERS):
            score += 2.0
            parts.append("cuda_context:+2.0")
        return score, parts
    if candidate.source == "cuda_perf" and has_any_marker(text, CUDA_CONTEXT_MARKERS):
        score += 2.0
        parts.append("cuda_source_context:+2.0")
    if candidate.source == "ml_infra" and has_any_marker(text, MODEL_CONTEXT_MARKERS):
        score += 2.0
        parts.append("ml_source_context:+2.0")
    if candidate.source in {"web_app", "python_packaging"}:
        score += 0.75
        parts.append(f"{candidate.source}:+0.75")
    return score, parts


def score_candidate(candidate: VocabCandidate, variant: str, text: str) -> tuple[float, tuple[str, ...]]:
    score = 0.5
    parts = ["variant_seen:+0.5"]
    if is_symbolic_exact_form(candidate.canonical):
        score += 2.0
        parts.append("symbolic_form:+2.0")
    if len(normalize_key(candidate.canonical)) >= 8:
        score += 0.75
        parts.append("long_form:+0.75")
    if candidate.source != "general_technical":
        score += 0.5
        parts.append(f"source_{candidate.source}:+0.5")
    context, context_parts = context_score(candidate, text)
    score += context
    parts.extend(context_parts)
    if candidate.canonical in AMBIGUOUS_CANONICALS:
        score -= 2.5
        parts.append("ambiguous_canonical:-2.5")
    if variant_is_risky(variant, candidate.canonical):
        score -= 1.0
        parts.append("risky_variant:-1.0")
    if variant.lower() == candidate.canonical.lower() and variant != candidate.canonical:
        score += 0.5
        parts.append("case_only_repair:+0.5")
    return score, tuple(parts)


def _replace_variant(text: str, variant: str, canonical: str) -> tuple[str, int]:
    if normalize_key(variant) == normalize_key(canonical) and variant == canonical:
        return text, 0
    if variant.lower() == canonical.lower() and canonical in text:
        return text, 0
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(variant) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    next_text, count = pattern.subn(canonical, text)
    if count:
        return next_text, count

    flexible_pattern = _flexible_symbolic_pattern(canonical)
    if flexible_pattern is None and re.search(r"[\s._/@=:+\-]", variant):
        flexible_pattern = _flexible_symbolic_pattern(variant)
    if flexible_pattern is None:
        return text, 0
    return flexible_pattern.subn(canonical, text)


def _flexible_symbolic_pattern(surface: str) -> re.Pattern[str] | None:
    chunks = re.findall(r"[A-Za-z0-9]+", surface)
    if len(chunks) < 2:
        return None
    body = FLEXIBLE_SEPARATOR_PATTERN.join(re.escape(chunk) for chunk in chunks)
    return re.compile(r"(?<![A-Za-z0-9_])" + body + r"(?![A-Za-z0-9_])", re.IGNORECASE)


class RepoVocabResolver:
    def __init__(self, candidates: Iterable[VocabCandidate]) -> None:
        self.candidates = tuple(candidates)
        by_key: dict[str, list[_VariantEntry]] = defaultdict(list)
        for candidate in self.candidates:
            for variant in candidate.variants:
                by_key[index_key(variant)].append(_VariantEntry(candidate, variant))
        self._variant_index = {
            key: tuple(sorted(entries, key=lambda entry: len(entry.variant), reverse=True))
            for key, entries in by_key.items()
        }

    @classmethod
    def from_vocab_files(cls, paths: Iterable[Path], extra_terms: Iterable[str] = ()) -> RepoVocabResolver:
        return cls(load_vocab_candidates(paths, extra_terms))

    def resolve_vocab(
        self,
        text: str,
        *,
        mode: ResolveMode = "gated",
        allowed_terms: set[str] | None = None,
        stage: str = "repo_vocab",
        threshold: float = 3.0,
    ) -> tuple[str, tuple[ResolverTrace, ...]]:
        corrected = str(text or "")
        traces: list[ResolverTrace] = []
        seen: set[tuple[str, str]] = set()
        active_entries: list[_VariantEntry] = []
        for key in text_index_keys(corrected):
            active_entries.extend(self._variant_index.get(key, ()))
        active_entries.sort(key=lambda entry: (entry.candidate.category, entry.candidate.canonical.lower(), -len(entry.variant)))
        for entry in active_entries:
            candidate = entry.candidate
            variant = entry.variant
            identity = (candidate.canonical, variant)
            if identity in seen:
                continue
            seen.add(identity)
            if allowed_terms is not None and candidate.canonical not in allowed_terms:
                continue
            candidate_score, score_parts = score_candidate(candidate, variant, corrected)
            if mode == "scored" and candidate_score < threshold:
                continue
            if mode == "gated" and not candidate_allowed_by_context(candidate, corrected):
                continue
            if mode == "gated" and variant_is_risky(variant, candidate.canonical) and not candidate_allowed_by_context(
                candidate, corrected
            ):
                continue
            if mode == "gated" and not is_symbolic_exact_form(candidate.canonical) and variant_is_risky(variant, candidate.canonical):
                continue
            next_text, count = _replace_variant(corrected, variant, candidate.canonical)
            if count:
                corrected = next_text
                traces.append(
                    ResolverTrace(
                        stage=stage,
                        canonical=candidate.canonical,
                        variant=variant,
                        category=candidate.category,
                        source=candidate.source,
                        reason=mode if mode != "scored" else f"score>={threshold}",
                        score=round(candidate_score, 4),
                        score_parts=score_parts,
                    )
                )
        return corrected, tuple(traces)

    def repair(
        self,
        text: str,
        *,
        mode: ResolveMode = "gated",
        allowed_terms: set[str] | None = None,
        apply_exact: bool = True,
        threshold: float = 3.0,
    ) -> RepairResult:
        raw = str(text or "")
        if apply_exact:
            exact_text, exact_rules = apply_exact_form_corrections(raw)
        else:
            exact_text, exact_rules = raw, []
        final, traces = self.resolve_vocab(exact_text, mode=mode, allowed_terms=allowed_terms, threshold=threshold)
        return RepairResult(
            raw_text=raw,
            exact_text=exact_text,
            final_text=final,
            exact_rules=tuple(exact_rules),
            vocab_traces=traces,
        )
