from src.models.schemas import ClinicalReport, HealthTimeline, PatientHistory, PatientState, RetrievedChunk, RiskPrediction


SYSTEM = """You are a cautious clinical information summarizer for a research prototype.
Use only supplied patient facts, risk outputs and retrieved passages. Never diagnose, prescribe,
or claim certainty. Return JSON with summary, risk_interpretation, suggested_next_steps,
precautions, questions_for_clinician, warning_signs, uncertainty, citations. Cite document IDs in square
brackets. Suggested steps must be general discussion points for a qualified clinician."""


class VerdictAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def reason(self, state: PatientState, history: PatientHistory, timeline: HealthTimeline,
               predictions: list[RiskPrediction], context: list[RetrievedChunk]) -> ClinicalReport:
        if self.llm is not None:
            try:
                data = self.llm.generate_json(SYSTEM, {
                    "patient": state.model_dump(mode="json"),
                    "separate_history": history.model_dump(mode="json"),
                    "timeline": timeline.model_dump(mode="json"),
                    "risk_outputs": [x.model_dump(mode="json") for x in predictions],
                    "retrieved_context": [x.model_dump(mode="json") for x in context],
                })
                return ClinicalReport.model_validate(data)
            except (ValueError, TypeError, RuntimeError):
                pass
        return self._fallback(state, history, timeline, predictions, context)

    @staticmethod
    def _fallback(state, history, timeline, predictions, context) -> ClinicalReport:
        relevant = [p for p in predictions if p.evidence != ["no encoded risk factor found"]]
        interpretations = [
            f"{p.condition}: {p.risk_band} prototype signal, based on {', '.join(p.evidence)}."
            for p in relevant
        ] or ["No Delphi score was available; interpretation is based on retrieved evidence only."]
        citations = [f"[{c.document_id}] {c.source}" for c in context]
        steps = ["Review the timeline and risk factors with a qualified clinician."]
        decision_chunks = [c for c in context if c.corpus == "decision"]
        steps.extend(c.text for c in decision_chunks[:2])
        return ClinicalReport(
            summary=(f"The timeline contains {len(timeline.events)} event(s) and has "
                     f"{timeline.data_quality} data quality. The main reported concern is "
                     f"{state.chief_complaint or 'not specified'}."),
            risk_interpretation=interpretations,
            suggested_next_steps=steps,
            precautions=[
                "Do not start, stop, or change medication based on this output.",
                "Confirm patient-reported history and model inputs with a qualified clinician.",
            ],
            questions_for_clinician=[
                "Which missing measurements or history would most change this assessment?",
                "Which validated risk calculator, if any, applies to this patient?",
            ],
            warning_signs=["Seek urgent help for severe breathing difficulty, chest pressure, stroke signs, loss of consciousness, or immediate self-harm risk."],
            uncertainty=("Delphi scores require a validated checkpoint; otherwise this is a retrieval-based summary. "
                         "Patient-reported history and retrieved passages may be incomplete."),
            citations=citations,
        )


ClinicalReasoningAgent = VerdictAgent
