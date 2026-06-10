from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ScopeSource = Literal["sticky", "query", "corpus", "doc_preretrieval"]


@dataclass(frozen=True)
class RetrievalScope:
    """Immutable per-turn retrieval scope threaded through the whole pipeline.

    None document_ids means the whole corpus (no filter).
    A non-empty tuple restricts retrieval to those doc_ids.
    An empty tuple is forbidden — IN [] would blank every query.
    """

    document_ids: tuple[str, ...] | None = None
    corpus_id: str | None = None
    rationale: str = "Whole corpus (no document restriction)."
    source: ScopeSource = "corpus"

    def __post_init__(self) -> None:
        if self.document_ids is not None and len(self.document_ids) == 0:
            raise ValueError(
                "RetrievalScope.document_ids must be None (whole corpus) or a non-empty tuple. "
                "An empty tuple would produce IN [] which matches zero rows."
            )

    @property
    def is_scoped(self) -> bool:
        """True when a specific set of documents is pinned."""
        return self.document_ids is not None

    @property
    def is_multi_doc(self) -> bool:
        """True when 2+ documents are in scope (enables cross-doc arms)."""
        return self.document_ids is None or len(self.document_ids) > 1

    @property
    def doc_id_list(self) -> list[str] | None:
        """Cypher-facing accessor: list or None (never empty list)."""
        return list(self.document_ids) if self.document_ids is not None else None

    @classmethod
    def whole_corpus(cls, corpus_id: str | None = None) -> "RetrievalScope":
        return cls(
            document_ids=None,
            corpus_id=corpus_id,
            rationale="Whole corpus (no document restriction).",
            source="corpus",
        )

    @classmethod
    def single(
        cls,
        doc_id: str,
        *,
        corpus_id: str | None = None,
        rationale: str = "",
        source: ScopeSource = "query",
    ) -> "RetrievalScope":
        return cls(
            document_ids=(doc_id,),
            corpus_id=corpus_id,
            rationale=rationale or f"Scoped to 1 document (doc_id={doc_id}).",
            source=source,
        )

    @classmethod
    def multi(
        cls,
        doc_ids: list[str] | tuple[str, ...],
        *,
        corpus_id: str | None = None,
        rationale: str = "",
        source: ScopeSource = "query",
    ) -> "RetrievalScope":
        ids = tuple(doc_ids)
        if not ids:
            raise ValueError("doc_ids must be non-empty; use whole_corpus() for no restriction.")
        return cls(
            document_ids=ids,
            corpus_id=corpus_id,
            rationale=rationale or f"Scoped to {len(ids)} documents.",
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_ids": self.doc_id_list,
            "corpus_id": self.corpus_id,
            "rationale": self.rationale,
            "source": self.source,
        }
