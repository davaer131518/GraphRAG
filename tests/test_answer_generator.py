from evidence.evidence_bundle import EvidenceBundle, EvidenceItem
from generation.answer_generator import AnswerGenerator


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, system: str, user: str, **kwargs) -> str:
        return self.response


def bundle() -> EvidenceBundle:
    item = EvidenceItem(
        block_id="p0017_b0000",
        type="paragraph",
        page=17,
        section_title="Risk Factors",
        section_path="Part I / Risk Factors",
        text="The App Store is subject to litigation and regulatory requirements.",
        retrieval_method="vector",
    )
    return EvidenceBundle(
        question="What does Apple say about App Store risks?",
        document_id=None,
        final_evidence=[item],
    )


def test_answer_generator_parses_valid_json() -> None:
    generator = AnswerGenerator(FakeLLM('{"answer":"A","confidence":"high","sources":[],"trace":["t"],"limitations":""}'))  # type: ignore[arg-type]
    answer = generator.generate(bundle())
    assert answer.answer == "A"
    assert answer.confidence == "high"
    # When LLM returns no sources, falls back to evidence items
    assert answer.sources[0].block_id == "p0017_b0000"
    assert answer.sources[0].section_title == "Risk Factors"


def test_answer_generator_falls_back_on_malformed_json() -> None:
    generator = AnswerGenerator(FakeLLM("not json"))  # type: ignore[arg-type]
    answer = generator.generate(bundle())
    assert answer.confidence == "low"
    assert "Structured answer generation failed" in answer.limitations
