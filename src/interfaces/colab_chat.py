from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from src.delphi.training import train as train_delphi
from src.llms.huggingface import load_llm
from src.models.schemas import PatientHistory, PatientState
from src.pipeline.clinical_pipeline import ClinicalPipeline
from src.pipeline.intake_pipeline import IntakePipeline


def ensure_demo_checkpoint(
    checkpoint: Path = Path("checkpoints/delphi-demo.pt"),
    epochs: int = 3,
) -> Path:
    """Create a tiny research-only checkpoint when no Delphi checkpoint is present."""
    if checkpoint.exists():
        return checkpoint
    print("Preparing the CPU Delphi demonstration checkpoint...")
    train_delphi(
        jsonl=Path("data/delphi/demo_trajectories.jsonl"),
        vocabulary=Path("data/delphi/vocabulary.json"),
        output=checkpoint,
        epochs=epochs,
    )
    return checkpoint


class ColabMedicalSession:
    """Stateful notebook interface backed by Qwen, Delphi and the four RAG stores."""

    def __init__(self, *, device: str = "cpu", train_demo_delphi: bool = True,
                 delphi_epochs: int = 3, debug: bool = True):
        checkpoint = None
        if train_demo_delphi:
            checkpoint = ensure_demo_checkpoint(epochs=delphi_epochs)
        self.settings = Settings(
            device=device,
            delphi_checkpoint=checkpoint,
            use_llm_reasoning=True,
        )
        self.llm = load_llm(self.settings)
        self.intake = IntakePipeline(self.llm, self.settings)
        self.clinical = ClinicalPipeline(self.settings, self.llm)
        self.state = PatientState()
        self.history = PatientHistory()
        self.messages: list[dict[str, str]] = []
        self.debug = debug

    def answer(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        result = self.intake.handle_turn(
            self.state, self.history, message, recent_messages=self.messages[-8:]
        )
        self.state, self.history = result.state, result.history
        self.messages.append({"role": "assistant", "content": result.reply})
        if self.debug:
            self.print_memory()
        return result.reply

    def print_memory(self) -> None:
        """Print the exact in-memory objects used by subsequent turns."""
        import json

        print("\n" + "-" * 72)
        print("DEBUG — CONVERSATION MEMORY")
        print(json.dumps(self.messages, ensure_ascii=False, indent=2))
        print("\nDEBUG — CURRENT PATIENT STATE")
        print(self.state.model_dump_json(indent=2))
        print("\nDEBUG — SEPARATE PATIENT HISTORY")
        print(self.history.model_dump_json(indent=2))
        print("-" * 72 + "\n")

    @property
    def ready(self) -> bool:
        return self.state.intake_complete and self.history.collection_complete

    def analyze(self):
        if not self.ready:
            raise RuntimeError("Complete the current complaint and history intake first.")
        return self.clinical.analyze(self.state, self.history)

    @staticmethod
    def print_report(result) -> None:
        report = result.report
        print("\n" + "=" * 72)
        print("FINAL RESEARCH VERDICT")
        print("=" * 72)
        print(f"Backend: {result.risk_backend}")
        print(result.model_notice)
        print(f"\nSummary\n{report.summary}")
        if result.predictions:
            print("\nDelphi-compatible next-event scores")
            for prediction in result.predictions:
                print(f"- {prediction.condition}: {prediction.score_0_to_1:.8f} "
                      f"({prediction.horizon})")
        else:
            print("\nNo Delphi score is available; this verdict is RAG-only.")
        print("\nInterpretation")
        for item in report.risk_interpretation:
            print(f"- {item}")
        print("\nPrecautions")
        for item in report.precautions:
            print(f"- {item}")
        print("\nQuestions / next steps")
        for item in report.suggested_next_steps + report.questions_for_clinician:
            print(f"- {item}")
        print("\nWarning signs")
        for item in report.warning_signs:
            print(f"- {item}")
        print("\nSources")
        for citation in report.citations:
            print(f"- {citation}")
        print(f"\nUncertainty\n{report.uncertainty}")
        print(f"\n{report.disclaimer}")


def run_interactive_session(session: ColabMedicalSession) -> None:
    print("Medical research prototype. Do not enter names or contact details.")
    print("Assistant: Describe your current concern in your own words.")
    while not session.ready:
        message = input("\nYou: ").strip()
        if not message:
            print("Assistant: Please enter a response.")
            continue
        reply = session.answer(message)
        print(f"Assistant: {reply}")
        if session.state.safety_requires_escalation:
            return
    print("\nRunning timeline, Delphi/RAG routing and the verdict model...")
    session.print_report(session.analyze())
