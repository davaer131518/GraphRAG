from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Confidence = Literal["high", "medium", "low"]


def make_snippet(text: str | None, max_chars: int = 360) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


@dataclass
class TraceStep:
    action: str
    description: str
    from_id: str | None = None
    to_id: str | None = None
    relationship: str | None = None
    method: str | None = None
    score: float | None = None
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    block_id: str
    type: str
    page: int | None
    text: str
    snippet: str = ""
    section_title: str | None = None      # populated from Section.title
    section_id: str | None = None          # holds Section.section_id
    section_path: str | None = None        # slash-delimited ancestry from KGBuilder
    section_level: int | None = None       # hierarchy depth
    doc_id: str | None = None              # source Document.doc_id
    doc_label: str | None = None           # human-friendly label (filename or doc_id)
    score: float | None = None
    retrieval_method: str = "unknown"
    relationship_path: list[str] = field(default_factory=list)
    source_relationship: str | None = None
    source_block_id: str | None = None
    why_relevant: str | None = None
    rank_features: dict[str, float] = field(default_factory=dict)
    mentioned_entities: list[dict[str, Any]] = field(default_factory=list)
    matched_entities: list[dict[str, Any]] = field(default_factory=list)
    relationship_confidence: float | None = None
    relationship_scope: str | None = None
    relationship_methods: list[str] = field(default_factory=list)
    table_html: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snippet:
            self.snippet = make_snippet(self.text)
        # Default doc_label to filename or doc_id
        if not self.doc_label and self.doc_id:
            self.doc_label = self.doc_id

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        retrieval_method: str,
        relationship_path: list[str] | None = None,
        source_relationship: str | None = None,
        source_block_id: str | None = None,
        why_relevant: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvidenceItem":
        doc_id = row.get("doc_id")
        doc_label = row.get("filename") or row.get("doc_label") or doc_id
        return cls(
            block_id=str(row.get("block_id") or row.get("id")),
            type=str(row.get("type") or "unknown"),
            page=row.get("page") or row.get("page_number"),
            text=row.get("text") or "",
            section_title=row.get("section_title") or row.get("section"),
            section_id=row.get("section_id"),
            section_path=row.get("section_path"),
            section_level=row.get("section_level"),
            doc_id=doc_id,
            doc_label=doc_label,
            score=row.get("score"),
            retrieval_method=retrieval_method,
            relationship_path=relationship_path or [retrieval_method],
            source_relationship=source_relationship,
            source_block_id=source_block_id,
            why_relevant=why_relevant,
            relationship_confidence=row.get("confidence"),
            relationship_scope=row.get("scope"),
            relationship_methods=list(row.get("methods") or []),
            table_html=row.get("table_html"),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    question: str
    document_ids: list[str] | None = None  # None = whole corpus; list = resolved scope
    corpus_id: str | None = None
    scope_rationale: str | None = None
    scope_source: str | None = None         # "sticky" | "query" | "corpus"
    answering_scope_note: str | None = None # post-retrieval: which doc(s) actually answered
    seed_blocks: list[EvidenceItem] = field(default_factory=list)
    expanded_blocks: list[EvidenceItem] = field(default_factory=list)
    final_evidence: list[EvidenceItem] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    ranking_debug: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def answering_doc_ids(self) -> list[str]:
        """The distinct doc_ids that appear in the final evidence."""
        return sorted({item.doc_id for item in self.final_evidence if item.doc_id})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCitation:
    page: int | None
    block_id: str
    type: str
    section_title: str | None
    section_path: str | None
    why_relevant: str
    snippet: str
    doc_id: str | None = None
    doc_label: str | None = None
    mentioned_entities: list[dict[str, Any]] = field(default_factory=list)
    table_html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalystAnswer:
    answer: str
    confidence: Confidence
    sources: list[SourceCitation]
    trace: list[str]
    limitations: str
    raw_evidence_bundle: EvidenceBundle
    raw_answer_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_evidence_bundle"] = self.raw_evidence_bundle.to_dict()
        return data


@dataclass
class TableRelationship:
    source_block_id: str
    source_page: int | None
    source_section: str | None
    source_snippet: str
    target_block_id: str
    target_page: int | None
    target_section: str | None
    target_snippet: str
    relation: str
    reason: str | None = None
    is_cross_doc: bool = False
    target_doc_id: str | None = None
    target_doc_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableExplorerResult:
    table_id: str
    relation_filter: str | None
    related_tables: list[TableRelationship]
    traces: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentMapResult:
    markdown: str
    sections: list[dict[str, Any]]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
