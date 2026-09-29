from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

import jiwer


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    return re.findall(r"[\w./+#-]+", text or "", flags=re.UNICODE)


def normalize_for_wer(text: str) -> str:
    transforms = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return transforms(text or "")


def normalize_words_for_wer(text: str) -> str:
    return " ".join(normalize_word_token(token) for token in words((text or "").lower()))


def normalize_word_token(token: str) -> str:
    # Preserve internal developer punctuation (`torch.compile`, `C++`, `.zshrc`,
    # `--version`) while ignoring sentence-final periods.
    while token.endswith(".") and len(token) > 1 and any(char.isalnum() for char in token[:-1]):
        token = token[:-1]
    return token


def compute_wer_cer(refs: list[str], hyps: list[str]) -> dict[str, float]:
    norm_refs = [normalize_for_wer(text) for text in refs]
    norm_hyps = [normalize_for_wer(text) for text in hyps]
    return {
        "wer": jiwer.wer(norm_refs, norm_hyps),
        "cer": jiwer.cer(norm_refs, norm_hyps),
    }


def compute_word_wer_cer(refs: list[str], hyps: list[str]) -> dict[str, float]:
    norm_refs = [normalize_words_for_wer(text) for text in refs]
    norm_hyps = [normalize_words_for_wer(text) for text in hyps]
    return {
        "wer": jiwer.wer(norm_refs, norm_hyps),
        "cer": jiwer.cer(norm_refs, norm_hyps),
    }


def technical_term_error_rate(term_lists: list[list[str]], hypotheses: list[str]) -> dict[str, float | int]:
    total = 0
    missed = 0
    for terms, hyp in zip(term_lists, hypotheses, strict=True):
        hyp_lower = hyp.lower()
        for term in terms:
            if not term:
                continue
            total += 1
            if term.lower() not in hyp_lower:
                missed += 1
    return {
        "technical_terms": total,
        "technical_term_misses": missed,
        "technical_term_error_rate": missed / total if total else 0.0,
    }


def casing_error_rate(refs: list[str], hyps: list[str]) -> float:
    total = 0
    errors = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_tokens = words(ref)
        hyp_counts = Counter(words(hyp))
        for token in ref_tokens:
            if not any(c.isupper() for c in token):
                continue
            total += 1
            if hyp_counts[token] <= 0:
                errors += 1
            else:
                hyp_counts[token] -= 1
    return errors / total if total else 0.0


def punctuation_error_rate(refs: list[str], hyps: list[str]) -> float:
    total = 0
    errors = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_punct = _PUNCT_RE.findall(ref or "")
        hyp_punct = _PUNCT_RE.findall(hyp or "")
        total += len(ref_punct)
        matcher = SequenceMatcher(a=ref_punct, b=hyp_punct)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        errors += max(0, len(ref_punct) - matched)
    return errors / total if total else 0.0


def hallucination_omission_counts(refs: list[str], hyps: list[str]) -> dict[str, int]:
    insertions = 0
    deletions = 0
    substitutions = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        matcher = SequenceMatcher(a=words(ref.lower()), b=words(hyp.lower()))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                insertions += j2 - j1
            elif tag == "delete":
                deletions += i2 - i1
            elif tag == "replace":
                substitutions += max(i2 - i1, j2 - j1)
    return {
        "hallucinated_words": insertions,
        "omitted_words": deletions,
        "substituted_words": substitutions,
    }
