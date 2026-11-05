"""Finnish morphology helpers for the multiif-fi verifier.

The English IFEval/MultiIF keyword checkers use plain regex/substring matching, which
under-scores correct Finnish because Finnish is heavily inflected (e.g. the keyword
"sota" would not match the inflected "sotaa"/"sodan"). These helpers add lemma-aware
matching via simplemma, plus Finnish sentence segmentation.

Used by multiif_fi.instructions to override keyword:existence / keyword:frequency /
keyword:forbidden_words checking and sentence counting.
"""
import re
import simplemma

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _lemma(token: str) -> str:
    t = token.lower()
    try:
        return simplemma.lemmatize(t, lang="fi")
    except Exception:
        return t


def lemmas(text: str) -> list[str]:
    """Lemmatised word tokens of `text` (Finnish)."""
    return [_lemma(tok) for tok in _WORD_RE.findall(text or "")]


def _target_lemmas(keyword: str) -> list[str]:
    """Lemmatise a (possibly multi-word) keyword; match requires all its lemmas present
    contiguously, but in practice keywords are single words."""
    return [_lemma(tok) for tok in _WORD_RE.findall(keyword or "")]


def keyword_count(text: str, keyword: str) -> int:
    """Count occurrences of `keyword` in `text` by lemma (Finnish-inflection aware).
    Falls back to case-insensitive substring count for non-word keywords (e.g. symbols)."""
    tl = _target_lemmas(keyword)
    if not tl:
        # non-word keyword: substring count
        return len(re.findall(re.escape(keyword), text or "", flags=re.IGNORECASE)) if keyword else 0
    toks = lemmas(text)
    if len(tl) == 1:
        return sum(1 for t in toks if t == tl[0])
    # multi-word: count contiguous lemma-sequence matches
    n = 0
    for i in range(len(toks) - len(tl) + 1):
        if toks[i:i + len(tl)] == tl:
            n += 1
    return n


def keyword_present(text: str, keyword: str) -> bool:
    return keyword_count(text, keyword) > 0


# ---- Finnish sentence segmentation -------------------------------------------------
_FI_ABBREV = {"esim", "ns", "mm", "ym", "yms", "jne", "ts", "vrt", "n", "nk", "ao",
              "ko", "ml", "mrd", "milj", "tri", "prof", "s", "v", "nro", "oy", "ry"}
_SENT_END = re.compile(r"[.!?…]+[\")'”’\]]*\s+")


def count_sentences_fi(text: str) -> int:
    """Count Finnish sentences. Uses NLTK's Finnish punkt tokenizer when available,
    else a punctuation heuristic that respects common Finnish abbreviations."""
    text = (text or "").strip()
    if not text:
        return 0
    try:
        import nltk
        try:
            tok = nltk.data.load("tokenizers/punkt/finnish.pickle")
        except Exception:
            nltk.download("punkt", quiet=True)
            tok = nltk.data.load("tokenizers/punkt/finnish.pickle")
        return len([s for s in tok.tokenize(text) if s.strip()])
    except Exception:
        # heuristic fallback
        parts = _SENT_END.split(text + " ")
        sents = [p for p in parts if p.strip() and p.strip().lower().rstrip(".") not in _FI_ABBREV]
        return max(1, len(sents))
