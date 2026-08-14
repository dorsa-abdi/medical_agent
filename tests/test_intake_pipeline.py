from config.settings import Settings
from src.models.schemas import PatientHistory, PatientState
from src.pipeline.intake_pipeline import IntakePipeline


class EmptyExtractionLLM:
    """Models the local LLM returning valid JSON without extracting any facts."""

    def generate_json(self, system: str, payload: dict) -> dict:
        return {}

    def generate(self, system: str, user: str) -> str:
        return "Please provide the requested information?"


def make_pipeline() -> IntakePipeline:
    return IntakePipeline(
        EmptyExtractionLLM(),
        Settings(question_timeout_seconds=1, max_follow_up_turns=12),
    )


def test_explicit_no_history_is_captured_when_llm_extraction_is_empty():
    pipeline = make_pipeline()

    first = pipeline.handle_turn(
        PatientState(), PatientHistory(), "I have had a headache since yesterday"
    )
    second = pipeline.handle_turn(
        first.state, first.history, "I have no relevant medical history"
    )

    assert second.history.explicitly_no_history is True
    assert second.history.collection_complete is True
    assert second.state.pending_field == "age"


def test_nothing_for_prior_medications_or_history_is_an_explicit_negative():
    pipeline = make_pipeline()

    first = pipeline.handle_turn(PatientState(), PatientHistory(), "I have a headache")
    second = pipeline.handle_turn(
        first.state, first.history, "nothing for prior medications or history"
    )

    assert second.history.explicitly_no_history is True
    assert second.history.collection_complete is True
    assert second.state.pending_field == "age"


def test_empty_history_extraction_does_not_advance_to_current_patient_questions():
    pipeline = make_pipeline()

    first = pipeline.handle_turn(PatientState(), PatientHistory(), "I have a headache")
    second = pipeline.handle_turn(first.state, first.history, "Nothing to add")

    assert second.history.collection_complete is False
    assert second.state.pending_field == "patient_history"


def test_question_answer_flow_collects_patient_data_without_llm_extraction():
    pipeline = make_pipeline()
    state, history = PatientState(), PatientHistory()

    messages = [
        "I have a headache",
        "nothing for prior medications or history",
        "42",
        "Since yesterday",
        "6 out of 10",
        "None",
    ]
    result = None
    for message in messages:
        result = pipeline.handle_turn(state, history, message)
        state, history = result.state, result.history

    assert result is not None
    assert result.status == "complete"
    assert state.intake_complete is True
    assert state.chief_complaint == "I have a headache"
    assert state.age == 42
    assert state.symptoms[0].onset == "Since yesterday"
    assert state.symptoms[0].severity_0_to_10 == 6
    assert state.associated_symptoms == ["None"]
