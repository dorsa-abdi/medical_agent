from src.models.schemas import ConversationMessage, PatientHistory, PatientState


SESSION_KEYS = ("patient_state", "patient_history", "messages", "analysis")


def initial_session_values() -> dict:
    return {
        "patient_state": PatientState(),
        "patient_history": PatientHistory(),
        "messages": [ConversationMessage(
            role="assistant",
            content="Describe your concern without names or contact details.",
        )],
        "analysis": None,
    }


def summary_is_available(state: PatientState) -> bool:
    return state.intake_complete and not state.safety_requires_escalation
