from src.interfaces.streamlit_state import (
    SESSION_KEYS,
    initial_session_values,
    summary_is_available,
)
from src.models.schemas import PatientState


def test_initial_session_values_are_fresh_and_contain_welcome_message():
    left = initial_session_values()
    right = initial_session_values()

    assert left["patient_state"] is not right["patient_state"]
    assert left["messages"][0].content.startswith("Describe your concern")
    assert left["analysis"] is None


def test_summary_is_available_only_after_nonurgent_completed_intake():
    assert summary_is_available(PatientState(intake_complete=True)) is True
    assert summary_is_available(PatientState(intake_complete=False)) is False
    assert summary_is_available(PatientState(
        intake_complete=True,
        safety_requires_escalation=True,
    )) is False


def test_session_keys_cover_all_resettable_state():
    assert SESSION_KEYS == (
        "patient_state", "patient_history", "messages", "analysis"
    )
