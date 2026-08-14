from typing import Protocol

from src.models.schemas import HistoryPatch, PatientHistory, PatientState, StatePatch


class JSONGenerator(Protocol):
    def generate_json(self, system: str, payload: dict) -> dict: ...


COLLECTOR_SYSTEM = """Extract only patient-stated facts into JSON; never infer a diagnosis.
Allowed fields: chief_complaint, symptoms, associated_symptoms, age, sex_at_birth,
pregnancy_possible, recent_events_or_exposures, relevant_negatives. Symptoms are objects with name,
onset, duration, severity_0_to_10, location, quality, pattern, aggravating_factors,
relieving_factors. Omit unknown scalars and use [] for unknown lists."""


class CollectorAgent:
    def __init__(self, llm: JSONGenerator):
        self.llm = llm

    def collect(self, state: PatientState, message: str,
                history: PatientHistory | None = None,
                recent_messages: list[dict] | None = None) -> StatePatch:
        data = self.llm.generate_json(
            COLLECTOR_SYSTEM,
            {
                "current_state": state.model_dump(mode="json"),
                "separate_patient_history": history.model_dump(mode="json") if history else None,
                "field_requested_in_previous_question": state.pending_field,
                "recent_conversation": recent_messages or [],
                "new_message": message,
            },
        )
        return StatePatch.model_validate(data)


HISTORY_SYSTEM = """Extract only explicitly stated historical health facts into JSON.
Return: conditions, previous_symptoms, medication_history, allergies, procedures,
family_history, lifestyle_factors, events, explicitly_no_history. An event has date
(YYYY-MM-DD), type, label, optional value/unit/status/source. Do not put the current
complaint into history. If the patient explicitly says there is no medical history,
set explicitly_no_history=true. Never infer diagnoses or dates."""


class HistoryCollectorAgent:
    def __init__(self, llm: JSONGenerator):
        self.llm = llm

    def collect(self, history: PatientHistory, message: str,
                state: PatientState | None = None,
                recent_messages: list[dict] | None = None) -> HistoryPatch:
        data = self.llm.generate_json(
            HISTORY_SYSTEM,
            {
                "current_history": history.model_dump(mode="json"),
                "current_encounter_for_context_only": state.model_dump(mode="json") if state else None,
                "recent_conversation": recent_messages or [],
                "new_message": message,
            },
        )
        return HistoryPatch.model_validate(data)
