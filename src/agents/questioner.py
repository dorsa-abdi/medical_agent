import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout


class QuestionerAgent:
    FALLBACKS = {
        "chief_complaint": "What is the main symptom or concern bothering you most?",
        "age": "How old are you?",
        "symptom_onset": "When did this start, and is it constant or intermittent?",
        "symptom_severity": "On a scale from 0 to 10, how severe is it now?",
        "associated_symptoms": "Do you have any other symptoms? If not, say none.",
    }
    HISTORY_FALLBACK = (
        "Before we continue, please describe your medical history separately: previous or "
        "ongoing diseases, medicines, allergies, procedures, family history, lifestyle factors "
        "and approximate dates. If there is no relevant history, say so explicitly."
    )

    def __init__(self, llm, timeout_seconds: float = 90.0):
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _as_question(text: str) -> str:
        question = " ".join(text.strip().strip('"').split())
        if question and question[-1] not in {"?", "؟"}:
            question += "?"
        return question

    def _fallback(self, fallback: str, reason: str) -> str:
        return f"[Fallback question — {reason}] {fallback}"

    def _generate(self, instruction: str, field: str, fallback: str) -> str:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="questioner")
        future = executor.submit(
            self.llm.generate,
            instruction,
            json.dumps({"next_missing_field": field}, ensure_ascii=False),
        )
        try:
            output = future.result(timeout=self.timeout_seconds)
            question = self._as_question(str(output))
            if question:
                return question[:500]
            return self._fallback(fallback, "Hugging Face returned an empty response")
        except FutureTimeout:
            future.cancel()
            return self._fallback(
                fallback, f"Hugging Face exceeded {self.timeout_seconds:g}s timeout"
            )
        except Exception as exc:
            return self._fallback(
                fallback, f"Hugging Face crashed: {type(exc).__name__}"
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def ask(self, next_missing_field: str) -> str:
        """Phrase exactly the field selected by ValidatorAgent."""
        return self._generate(
            "You are a medical intake questioner. Ask exactly one short, neutral question "
            "that obtains the supplied {next_missing_field}. Do not select another field, "
            "diagnose, recommend treatment, or request identifying information. Return only "
            "the question.",
            next_missing_field,
            self.FALLBACKS[next_missing_field],
        )

    def ask_history(self) -> str:
        return self._generate(
            "Ask exactly one concise question for the supplied next_missing_field. Request "
            "previous and ongoing diseases, medicines, allergies, procedures, family history, "
            "lifestyle factors and approximate dates. Allow the patient to explicitly report "
            "no relevant history. Return only the question.",
            "patient_history",
            self.HISTORY_FALLBACK,
        )
