from config.settings import Settings
from src.agents.clinical_reasoner import VerdictAgent
from src.delphi.predictor import DelphiPredictor
from src.models.schemas import AnalysisResult, PatientHistory, PatientState
from src.pipeline.timeline import TimelineBuilder
from src.rag.store import MultiRAG


class ClinicalPipeline:
    """Routes analysis through Delphi+RAG or RAG-only, then asks the verdict LLM."""

    def __init__(self, settings: Settings, llm=None, rag: MultiRAG | None = None):
        self.settings = settings
        self.timeline_builder = TimelineBuilder()
        self.rag = rag or MultiRAG.from_directory(settings.knowledge_dir)
        self.verdict = VerdictAgent(llm if settings.use_llm_reasoning else None)
        self.delphi = (
            DelphiPredictor(settings.delphi_checkpoint, settings.delphi_vocabulary,
                            settings.delphi_horizon_years)
            if settings.delphi_checkpoint else None
        )

    @staticmethod
    def _patient_memory(history: PatientHistory) -> list[dict]:
        documents = []
        for event in history.events:
            documents.append({
                "id": f"patient-{event.id}", "title": f"Patient event: {event.label}",
                "text": f"{event.date.isoformat()} {event.type}: {event.label}; status={event.status}",
                "source": event.source,
            })
        groups = {
            "conditions": history.conditions, "medications": history.medication_history,
            "allergies": history.allergies, "family history": history.family_history,
            "lifestyle": history.lifestyle_factors, "procedures": history.procedures,
        }
        for group, values in groups.items():
            if values:
                documents.append({
                    "id": f"patient-{group.replace(' ', '-')}", "title": f"Patient {group}",
                    "text": "; ".join(values), "source": "patient_reported",
                })
        return documents

    def analyze(self, state: PatientState, history: PatientHistory) -> AnalysisResult:
        timeline = self.timeline_builder.build(state, history)
        predictions = []
        backend = "rag_only"
        notice = "No usable Delphi checkpoint; analysis uses retrieved evidence only."
        if self.delphi is not None and self.delphi.available and state.age is not None and timeline.events:
            predictions = self.delphi.predict(timeline, state)
            if predictions:
                backend = "delphi"
                notice = "Delphi-compatible next-event scores were included in evidence retrieval and verdict generation."

        query = " ".join(filter(None, [
            state.chief_complaint or "",
            " ".join(symptom.name for symptom in state.symptoms),
            " ".join(history.conditions + history.family_history + history.lifestyle_factors),
            " ".join(prediction.condition for prediction in predictions),
        ]))
        context = self.rag.retrieve(
            query, self.settings.top_k, patient_documents=self._patient_memory(history)
        )
        report = self.verdict.reason(state, history, timeline, predictions, context)
        return AnalysisResult(
            timeline=timeline, predictions=predictions, retrieved_context=context,
            report=report, used_longitudinal_model=backend == "delphi",
            risk_backend=backend, model_notice=notice,
        )
