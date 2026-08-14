import re
from dataclasses import dataclass

from src.models.schemas import PatientState, StatePatch, Symptom, ValidationResult


@dataclass(frozen=True)
class AnswerCapture:
    accepted: bool
    patch: StatePatch
    guidance: str | None = None


class ValidatorAgent:
    """Checks completeness and independently captures the specifically requested answer."""

    FIELD_LABELS = {
        "chief_complaint": "main concern",
        "age": "age",
        "symptom_onset": "when the main symptom started",
        "symptom_severity": "severity from 0 to 10",
        "associated_symptoms": "other symptoms or an explicit statement that there are none",
    }
    ANSWER_GUIDANCE = {
        "chief_complaint": "Describe the symptom or concern in a short sentence.",
        "age": "Enter your age as a number between 0 and 125, for example: 42.",
        "symptom_onset": "Say when it began, for example: three days ago or in May.",
        "symptom_severity": "Enter one number from 0 to 10, for example: 6/10.",
        "associated_symptoms": "List other symptoms, or write: none.",
    }

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.split()).strip()

    def capture_requested_answer(self, state: PatientState, field: str | None,
                                 user_text: str) -> AnswerCapture:
        text = self._clean(user_text)
        if not field or field == "patient_history":
            return AnswerCapture(False, StatePatch())
        if not text:
            return AnswerCapture(False, StatePatch(), self.ANSWER_GUIDANCE.get(field))

        primary = state.symptoms[0] if state.symptoms else None
        if field == "chief_complaint":
            return AnswerCapture(True, StatePatch(
                chief_complaint=text[:500], symptoms=[Symptom(name=text[:200])]
            ))
        if field == "age":
            match = re.search(r"(?<!\d)(1(?:[01]\d|2[0-5])|[1-9]?\d)(?!\d)", text)
            if match:
                return AnswerCapture(True, StatePatch(age=int(match.group(1))))
        elif field == "symptom_onset" and primary:
            return AnswerCapture(True, StatePatch(symptoms=[Symptom(
                name=primary.name, onset=text[:300]
            )]))
        elif field == "symptom_severity" and primary:
            match = re.search(r"(?<!\d)(10|[0-9])(?:\s*/\s*10)?(?!\d)", text)
            words = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            }
            word_match = next((value for word, value in words.items()
                               if re.search(rf"\b{word}\b", text.casefold())), None)
            if match or word_match is not None:
                severity = int(match.group(1)) if match else word_match
                return AnswerCapture(True, StatePatch(symptoms=[Symptom(
                    name=primary.name, severity_0_to_10=severity
                )]))
        elif field == "associated_symptoms":
            return AnswerCapture(True, StatePatch(associated_symptoms=[text[:500]]))
        return AnswerCapture(False, StatePatch(), self.ANSWER_GUIDANCE.get(field))

    def field_is_present(self, state: PatientState, field: str | None) -> bool:
        if field == "chief_complaint":
            return bool(state.chief_complaint and state.symptoms)
        if field == "age":
            return state.age is not None
        if field == "symptom_onset":
            return bool(state.symptoms and (state.symptoms[0].onset or state.symptoms[0].duration))
        if field == "symptom_severity":
            return bool(state.symptoms and state.symptoms[0].severity_0_to_10 is not None)
        if field == "associated_symptoms":
            return bool(state.associated_symptoms)
        return False

    def validate(self, state: PatientState) -> ValidationResult:
        missing = []
        if not state.chief_complaint or not state.symptoms:
            missing.append("chief_complaint")
        if state.age is None:
            missing.append("age")
        primary = state.symptoms[0] if state.symptoms else None
        if primary:
            if not primary.onset and not primary.duration:
                missing.append("symptom_onset")
            if primary.severity_0_to_10 is None:
                missing.append("symptom_severity")
        if not state.associated_symptoms:
            missing.append("associated_symptoms")

        missing = [
            field for field in dict.fromkeys(missing)
            if field not in state.unavailable_fields
        ]
        total = len(self.FIELD_LABELS)
        return ValidationResult(
            missing_fields=[self.FIELD_LABELS[item] for item in missing],
            completion_score=max(0.0, (total - len(missing)) / total),
            is_sufficient=not missing,
            next_field=missing[0] if missing else None,
        )
