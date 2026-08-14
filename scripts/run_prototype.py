import argparse
import json
from pathlib import Path

from config.settings import Settings
from src.models.schemas import PatientHistory, PatientState, TimelineEvent
from src.pipeline.clinical_pipeline import ClinicalPipeline


def main():
    parser = argparse.ArgumentParser(description="Run phase 2 without loading an LLM")
    parser.add_argument("--timeline", type=Path, default=Path("examples/timeline.json"))
    args = parser.parse_args()
    events = [TimelineEvent.model_validate(item) for item in json.loads(args.timeline.read_text())]
    state = PatientState(
        chief_complaint="preventive risk review", age=57, intake_complete=True,
    )
    history = PatientHistory(
        conditions=["hypertension"], lifestyle_factors=["smoking"],
        events=events, collection_complete=True,
    )
    settings = Settings(use_llm_reasoning=False)
    result = ClinicalPipeline(settings=settings).analyze(state, history)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
