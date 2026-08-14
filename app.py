import logging

import streamlit as st

from config.settings import get_settings
from src.interfaces.streamlit_state import (
    SESSION_KEYS,
    initial_session_values,
    summary_is_available,
)
from src.llms.huggingface import load_llm
from src.models.schemas import ConversationMessage
from src.pipeline.clinical_pipeline import ClinicalPipeline
from src.pipeline.intake_pipeline import IntakePipeline

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Medical Timeline Prototype", layout="wide")


@st.cache_resource(show_spinner="Loading pipeline (first run may download the model)...")
def get_pipelines():
    settings = get_settings()
    llm = load_llm(settings)
    return IntakePipeline(llm, settings), ClinicalPipeline(settings, llm)


def initialize():
    for key, value in initial_session_values().items():
        st.session_state.setdefault(key, value)


def reset():
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)
    st.rerun()


def render_report(result):
    report = result.report
    st.warning(report.disclaimer)
    st.subheader("Clinical information summary")
    st.write(report.summary)
    st.subheader("Prototype risk signals")
    st.info(f"Backend: {result.risk_backend}. {result.model_notice}")
    if result.predictions:
        st.dataframe([p.model_dump() for p in result.predictions], use_container_width=True)
    else:
        st.info("No Delphi score is available; the verdict uses RAG evidence only.")
    st.subheader("Interpretation")
    for item in report.risk_interpretation:
        st.write(f"- {item}")
    st.subheader("Points to discuss with a clinician")
    for item in report.suggested_next_steps:
        st.write(f"- {item}")
    st.subheader("Precautions")
    for item in report.precautions:
        st.write(f"- {item}")
    with st.expander("Timeline and retrieved evidence"):
        st.json(result.timeline.model_dump(mode="json"))
        for chunk in result.retrieved_context:
            st.markdown(f"**[{chunk.document_id}] {chunk.title}** — {chunk.source}")
            st.write(chunk.text)
    st.caption(report.uncertainty)


def main():
    initialize()
    st.title("Medical Agent Chat")
    st.warning(
        "Research prototype only. Do not enter identifying information. "
        "This is not a diagnosis or a substitute for professional care."
    )
    intake, clinical = get_pipelines()

    state = st.session_state.patient_state
    progress, action = st.columns([5, 1])
    with progress:
        st.progress(state.completion_score)
        st.caption(f"Intake {state.completion_score:.0%} complete")
    with action:
        if st.button("Reset", use_container_width=True):
            reset()

    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.write(message.content)

    if user_text := st.chat_input("Describe symptoms or answer the question"):
        st.session_state.messages.append(ConversationMessage(role="user", content=user_text))
        try:
            result = intake.handle_turn(
                st.session_state.patient_state,
                st.session_state.patient_history,
                user_text,
                recent_messages=[
                    message.model_dump(mode="json")
                    for message in st.session_state.messages[-8:]
                ],
            )
            st.session_state.patient_state = result.state
            st.session_state.patient_history = result.history
            st.session_state.messages.append(ConversationMessage(
                role="assistant", content=result.reply
            ))
            st.rerun()
        except Exception as exc:
            logger.exception("Intake failed")
            st.error(f"Could not process the message: {exc}")

    if summary_is_available(st.session_state.patient_state):
        if st.button("Generate summary", type="primary"):
            try:
                with st.spinner("Generating summary..."):
                    st.session_state.analysis = clinical.analyze(
                        st.session_state.patient_state,
                        st.session_state.patient_history,
                    )
            except Exception as exc:
                logger.exception("Summary generation failed")
                st.error(f"Could not generate the summary: {exc}")

    if st.session_state.analysis is not None:
        st.divider()
        with st.container():
            st.header("Clinical summary")
            render_report(st.session_state.analysis)


if __name__ == "__main__":
    main()
