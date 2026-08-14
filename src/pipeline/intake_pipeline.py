import re

from config.settings import Settings
from src.agents.collector import CollectorAgent, HistoryCollectorAgent
from src.agents.questioner import QuestionerAgent
from src.agents.safety import SafetyAgent
from src.agents.validator import ValidatorAgent
from src.models.history_manager import apply_history_patch
from src.models.schemas import PatientHistory, PatientState, TurnResult
from src.models.schemas import StatePatch, Symptom
from src.models.state_manager import apply_patch


class IntakePipeline:
    def __init__(self, llm, settings: Settings):
        self.current_collector = CollectorAgent(llm)
        self.history_collector = HistoryCollectorAgent(llm)
        self.validator = ValidatorAgent()
        self.questioner = QuestionerAgent(llm, settings.question_timeout_seconds)
        self.safety = SafetyAgent()
        self.settings = settings

    @staticmethod
    def _preserve_first_complaint(state: PatientState, message: str) -> PatientState:
        """Preserve explicit patient text when a small LLM cannot emit valid JSON."""
        complaint = " ".join(message.split()).strip()
        if not complaint:
            return state
        patch = StatePatch(
            chief_complaint=complaint[:500],
            symptoms=[Symptom(name=complaint[:200])],
        )
        return apply_patch(state, patch)

    @staticmethod
    def _explicitly_declines_history(message: str) -> bool:
        """Recognize a direct negative answer without depending on LLM extraction."""
        text = " ".join(message.casefold().split()).strip(" .!?,;:")
        if text in {"no", "none", "nothing", "not applicable", "n/a"}:
            return True
        if re.search(
            r"\bnothing\s+(?:to\s+report\s+)?(?:for|regarding|about)\b.*"
            r"\b(?:medications?|medicines?|conditions?|allerg(?:y|ies)|histor(?:y|ies))\b",
            text,
        ):
            return True
        return bool(re.search(
            r"\b(?:no|don't have|do not have|without)\b.*"
            r"\b(?:medical|health|relevant|past)\s+histor(?:y|ies)\b",
            text,
        ))

    def handle_turn(self, state: PatientState, history: PatientHistory,
                    user_message: str,
                    recent_messages: list[dict] | None = None) -> TurnResult:
        current = state.model_copy(deep=True)
        longitudinal = history.model_copy(deep=True)
        requested_field = current.pending_field
        current.turn_count += 1

        flags = self.safety.screen(user_message)
        if flags:
            current.intake_complete = False
            current.safety_level = "emergency"
            current.safety_reason = ", ".join(flags)
            current.safety_requires_escalation = True
            return TurnResult(
                state=current, history=longitudinal, status="urgent",
                reply=(f"This may be an emergency ({', '.join(flags)}). Stop this chat and "
                       f"contact {self.settings.emergency_number} now. Do not drive yourself."),
            )

        if current.turn_count == 1 and not longitudinal.collection_complete:
            try:
                current = apply_patch(
                    current, self.current_collector.collect(
                        current, user_message, longitudinal, recent_messages
                    )
                )
            except (ValueError, TypeError, RuntimeError):
                current = self._preserve_first_complaint(current, user_message)
            if not current.chief_complaint:
                current = self._preserve_first_complaint(current, user_message)
            current.pending_field = "patient_history"
            return TurnResult(
                state=current, history=longitudinal, status="collecting",
                reply=self.questioner.ask_history(),
            )

        if not longitudinal.collection_complete:
            if self._explicitly_declines_history(user_message):
                longitudinal.explicitly_no_history = True
                longitudinal.collection_complete = True
            else:
                try:
                    patch = self.history_collector.collect(
                        longitudinal, user_message, current, recent_messages
                    )
                    longitudinal = apply_history_patch(longitudinal, patch)
                except (ValueError, TypeError, RuntimeError):
                    return TurnResult(
                        state=current, history=longitudinal, status="collecting",
                        reply="I could not structure that history. Please list prior conditions or say ‘no relevant history’.",
                    )
            if not longitudinal.collection_complete:
                current.pending_field = "patient_history"
                return TurnResult(
                    state=current, history=longitudinal, status="collecting",
                    reply=("I could not record any medical-history details. Please list prior "
                           "conditions, medicines, allergies, or say ‘no relevant history’."),
                )

        else:
            try:
                current = apply_patch(
                    current, self.current_collector.collect(
                        current, user_message, longitudinal, recent_messages
                    )
                )
            except (ValueError, TypeError, RuntimeError):
                if not current.chief_complaint:
                    current = self._preserve_first_complaint(current, user_message)
            if not current.chief_complaint:
                current = self._preserve_first_complaint(current, user_message)

        answer_accepted = requested_field == "patient_history" and longitudinal.collection_complete
        answer_guidance = None
        if requested_field and requested_field != "patient_history":
            skipped = user_message.strip().casefold() in {"skip", "unknown", "i don't know", "i do not know"}
            if skipped and requested_field not in current.unavailable_fields:
                current.unavailable_fields.append(requested_field)
            answer_accepted = skipped or self.validator.field_is_present(current, requested_field)
            if not answer_accepted:
                capture = self.validator.capture_requested_answer(
                    current, requested_field, user_message
                )
                answer_guidance = capture.guidance
                if capture.accepted:
                    current = apply_patch(current, capture.patch)
                    answer_accepted = self.validator.field_is_present(current, requested_field)

        validation = self.validator.validate(current)
        current.missing_fields = validation.missing_fields
        current.completion_score = validation.completion_score
        current.intake_complete = validation.is_sufficient
        if validation.is_sufficient:
            current.pending_field = None
            current.pending_attempts = 0
            return TurnResult(state=current, history=longitudinal, status="complete",
                              reply="The current concern and medical history are ready for analysis.")
        if current.turn_count >= self.settings.max_follow_up_turns:
            return TurnResult(
                state=current, history=longitudinal, status="complete",
                reply="Automated intake stopped. Missing: " + ", ".join(current.missing_fields),
            )
        next_field = validation.next_field
        repeated_unaccepted = (
            requested_field is not None
            and requested_field == next_field
            and not answer_accepted
        )
        if repeated_unaccepted:
            current.pending_attempts += 1
        else:
            current.pending_attempts = 0
        current.pending_field = next_field
        question = self.questioner.ask(next_field)
        if repeated_unaccepted:
            label = self.validator.FIELD_LABELS[next_field]
            guidance = answer_guidance or self.validator.ANSWER_GUIDANCE[next_field]
            if current.pending_attempts > 1:
                question = (
                    f"I still could not record your answer for: {label}. {guidance} "
                    "Please rephrase it, or type ‘skip’ if you do not know."
                )
            else:
                question = (
                    f"I received your message but could not record it as {label}. "
                    f"{guidance}\n\n{question}"
                )
        return TurnResult(
            state=current, history=longitudinal, status="collecting",
            reply=question,
        )
