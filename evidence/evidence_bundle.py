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
    section: str | None = None
    section_block_id: str | None = None
    score: float | None = None
    retrieval_method: str = "unknown"
    relationship_path: list[str] = field(default_factory=list)
    source_relationship: str | None = None
    source_block_id: str | None = None
    why_relevant: str | None = None
    rank_features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snippet:
            self.snippet = make_snippet(self.text)

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
        return cls(
            block_id=str(row.get("block_id") or row.get("id")),
            type=str(row.get("type") or "unknown"),
            page=row.get("page") or row.get("page_number"),
            text=row.get("text") or "",
            section=row.get("section"),
            section_block_id=row.get("section_id"),
            score=row.get("score"),
            retrieval_method=retrieval_method,
            relationship_path=relationship_path or [retrieval_method],
            source_relationship=source_relationship,
            source_block_id=source_block_id,
            why_relevant=why_relevant,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    question: str
    document_id: str | None
    seed_blocks: list[EvidenceItem] = field(default_factory=list)
    expanded_blocks: list[EvidenceItem] = field(default_factory=list)
    final_evidence: list[EvidenceItem] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    ranking_debug: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCitation:
    page: int | None
    block_id: str
    type: str
    section: str | None
    why_relevant: str
    snippet: str

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
