"""Document scope resolver.

Deterministic only: map explicit document/version references in the question to doc_ids
using Document metadata (filename stem, logical_doc_key, doc_family, version_id,
published_at year). Fast, fully serviceless-testable, precision-safe.

Ambiguous or unrecognised references fall back to whole corpus — never guessing a single
document and risking exclusion of the answer. The real document selection happens
post-retrieval: the ranker surfaces the most relevant blocks regardless of document, and
per-document attribution in the answer shows which document(s) actually answered.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from retrievers.scope import RetrievalScope

logger = logging.getLogger(__name__)

# Regex to extract 4-digit years
_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")

# Comparison-intent cues (case-insensitive)
_COMPARISON_CUES_RE = re.compile(
    r"\b(vs\.?|versus|compare[sd]?(\s+to)?|difference\s+between|across\s+(years|filings|documents|versions))\b",
    re.IGNORECASE,
)

# Generic families that shouldn't trigger single-family scoping
_GENERIC_FAMILIES = frozenset({"default", "unknown", "report", "reports", "document", "documents", ""})


class _DeterministicResult(Enum):
    CONFIDENT_SINGLE = "confident_single"
    CONFIDENT_SET = "confident_set"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass
class _Determination:
    result: _DeterministicResult
    matched_doc_ids: list[str]
    rationale: str


def _normalize(s: str) -> str:
    """Lowercase and collapse non-alphanumeric chars to spaces."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _filename_stem(filename: str) -> str:
    """Strip directory and extension from a filename."""
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem


def _extract_years(question: str) -> set[str]:
    return set(_YEAR_RE.findall(question))


def _has_comparison_cue(question: str) -> bool:
    return bool(_COMPARISON_CUES_RE.search(question))


def _matches_doc(question_norm: str, question_years: set[str], doc: dict[str, Any]) -> bool:
    """Return True if any metadata field of the doc has a confident match in the question."""
    # Filename stem (min length 4 to avoid short false hits)
    fname = doc.get("filename") or ""
    if fname:
        stem = _normalize(_filename_stem(fname))
        if len(stem) >= 4 and stem in question_norm:
            return True

    # logical_doc_key (highest precision — reliable grouping key)
    ldk = _normalize(doc.get("logical_doc_key") or "")
    if len(ldk) >= 4 and ldk in question_norm:
        return True

    # version_id (exact substring)
    vid = _normalize(doc.get("version_id") or "")
    if len(vid) >= 2 and vid in question_norm:
        return True

    # published_at year
    pub = doc.get("published_at") or ""
    if pub:
        pub_year = str(pub)[:4]
        if pub_year in question_years:
            return True

    return False


def _family_matches(question_norm: str, doc: dict[str, Any]) -> str | None:
    """Return the normalized doc_family if it matches the question (and is non-generic)."""
    fam = doc.get("doc_family") or ""
    fam_norm = _normalize(fam)
    if fam_norm in _GENERIC_FAMILIES:
        return None
    if len(fam_norm) >= 4 and fam_norm in question_norm:
        return fam_norm
    return None


def determine_scope(question: str, documents: list[dict[str, Any]]) -> _Determination:
    """Deterministic core — pure function, serviceless, no LLM."""
    if not documents:
        return _Determination(
            result=_DeterministicResult.NONE,
            matched_doc_ids=[],
            rationale="Whole corpus (no documents in corpus).",
        )

    q_norm = _normalize(question)
    q_years = _extract_years(question)
    has_cue = _has_comparison_cue(question)

    # Direct metadata matches (filename, logical_doc_key, version_id, published_at year)
    direct_matches = [doc for doc in documents if _matches_doc(q_norm, q_years, doc)]

    # Family matches (low precision, yields a group)
    family_groups: dict[str, list[dict]] = {}
    for doc in documents:
        fam = _family_matches(q_norm, doc)
        if fam:
            family_groups.setdefault(fam, []).append(doc)

    # Combine: prefer direct matches over pure family matches
    all_matched_ids: list[str] = []
    seen: set[str] = set()
    for doc in direct_matches:
        did = doc["doc_id"]
        if did not in seen:
            all_matched_ids.append(did)
            seen.add(did)
    # Add family matches that weren't already captured
    for fam_docs in family_groups.values():
        for doc in fam_docs:
            did = doc["doc_id"]
            if did not in seen:
                all_matched_ids.append(did)
                seen.add(did)

    n = len(all_matched_ids)

    if n == 0:
        return _Determination(
            result=_DeterministicResult.NONE,
            matched_doc_ids=[],
            rationale="Whole corpus (no confident document reference in query).",
        )

    if n == 1:
        doc = next(d for d in documents if d["doc_id"] == all_matched_ids[0])
        fname = doc.get("filename") or all_matched_ids[0]
        return _Determination(
            result=_DeterministicResult.CONFIDENT_SINGLE,
            matched_doc_ids=all_matched_ids,
            rationale=f"Scoped to 1 document by query cue: '{fname}' (doc_id={all_matched_ids[0]}).",
        )

    # Multiple matches with a comparison cue → confident set
    if has_cue:
        fnames = ", ".join(
            f"'{d.get('filename') or d['doc_id']}'"
            for d in documents
            if d["doc_id"] in all_matched_ids
        )
        return _Determination(
            result=_DeterministicResult.CONFIDENT_SET,
            matched_doc_ids=all_matched_ids,
            rationale=(
                f"Scoped to {n} documents by query cue: {fnames}; "
                f"comparison intent detected."
            ),
        )

    # Multiple matches without a comparison cue → ambiguous (LLM may confirm, else whole corpus)
    fnames = ", ".join(
        f"'{d.get('filename') or d['doc_id']}'"
        for d in documents
        if d["doc_id"] in all_matched_ids
    )
    return _Determination(
        result=_DeterministicResult.AMBIGUOUS,
        matched_doc_ids=all_matched_ids,
        rationale=(
            f"Whole corpus (reference matched {n} documents ambiguously [{fnames}]; "
            f"not narrowing to avoid excluding the answer)."
        ),
    )


class ScopeResolver:
    """Deterministic document scope resolver.

    Maps explicit document/version references in the question to doc_ids via metadata
    matching. Ambiguous or unrecognised references fall back to whole corpus — the ranker
    and per-document attribution handle implicit document selection post-retrieval.
    """

    def __init__(self, settings: Any, llm: Any | None = None) -> None:
        self.settings = settings
        # llm param kept for API compatibility but not used

    def resolve(
        self,
        question: str,
        documents: list[dict[str, Any]],
        corpus_id: str | None = None,
    ) -> RetrievalScope:
        det = determine_scope(question, documents)

        if det.result == _DeterministicResult.CONFIDENT_SINGLE:
            return RetrievalScope.single(
                det.matched_doc_ids[0],
                corpus_id=corpus_id,
                rationale=det.rationale,
                source="query",
            )

        if det.result == _DeterministicResult.CONFIDENT_SET:
            return RetrievalScope.multi(
                det.matched_doc_ids,
                corpus_id=corpus_id,
                rationale=det.rationale,
                source="query",
            )

        # AMBIGUOUS or NONE → whole corpus (precision-safe; ranker handles implicit selection)
        return RetrievalScope.whole_corpus(corpus_id=corpus_id)._replace_rationale(det.rationale)


def _make_scope_whole_corpus_with_rationale(rationale: str, corpus_id: str | None) -> RetrievalScope:
    """Helper so we can attach a rationale to a whole-corpus scope without dataclass mutation."""
    return RetrievalScope(
        document_ids=None,
        corpus_id=corpus_id,
        rationale=rationale,
        source="corpus",
    )


# Monkey-patch a helper onto RetrievalScope so ScopeResolver.resolve can attach a rationale
# to a whole-corpus scope without breaking the frozen dataclass.
def _replace_rationale(self: RetrievalScope, rationale: str) -> RetrievalScope:
    return RetrievalScope(
        document_ids=self.document_ids,
        corpus_id=self.corpus_id,
        rationale=rationale,
        source=self.source,
    )


RetrievalScope._replace_rationale = _replace_rationale  # type: ignore[attr-defined]
