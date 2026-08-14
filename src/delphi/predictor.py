import json
from datetime import date
from pathlib import Path

from src.delphi.model import DelphiConfig, build_model
from src.models.schemas import HealthTimeline, PatientState, RiskPrediction


class DelphiPredictor:
    """Inference adapter for locally trained or authorized Delphi checkpoints."""

    def __init__(self, checkpoint: Path, vocabulary: Path, horizon_years: float = 5):
        self.checkpoint_path = Path(checkpoint)
        self.vocabulary_path = Path(vocabulary)
        self.horizon_years = horizon_years
        self.model = None
        self.labels: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.predictable_ids: set[int] = set()

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists() and self.vocabulary_path.exists()

    def _load(self):
        if self.model is not None:
            return
        if not self.available:
            raise FileNotFoundError("Delphi checkpoint or vocabulary is missing")
        import torch

        vocab = json.loads(self.vocabulary_path.read_text(encoding="utf-8"))
        self.labels = vocab["labels"]
        self.token_to_id = {label.casefold(): i for i, label in enumerate(self.labels)}
        predictable = vocab.get("predictable_labels", self.labels[1:])
        self.predictable_ids = {
            self.token_to_id[label.casefold()] for label in predictable
            if label.casefold() in self.token_to_id
        }
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        config = DelphiConfig(**checkpoint["config"])
        if len(self.labels) != config.vocab_size:
            raise ValueError("Vocabulary size does not match the checkpoint")
        self.model = build_model(config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def predict(self, timeline: HealthTimeline, state: PatientState, top_k: int = 10) -> list[RiskPrediction]:
        self._load()
        import torch

        if state.age is None:
            return []
        today = date.today()
        encoded = []
        for event in timeline.events:
            token = self.token_to_id.get(event.label.casefold())
            if token is None or token == 0:
                continue
            event_age = max(0.0, state.age * 365.25 - (today - event.date).days)
            encoded.append((event.date, token, event_age))
        if not encoded:
            return []
        encoded = encoded[-self.model.config.block_size:]
        ids = torch.tensor([[item[1] for item in encoded]], dtype=torch.long)
        ages = torch.tensor([[item[2] for item in encoded]], dtype=torch.float32)
        probs = self.model.horizon_probabilities(ids, ages, self.horizon_years * 365.25)[0]
        observed = set(ids[0].tolist())
        allowed = self.predictable_ids - observed
        candidates = sorted(allowed, key=lambda index: float(probs[index]), reverse=True)[:top_k]
        results = []
        for index in candidates:
            score = float(probs[index])
            band = "high" if score >= 0.67 else "moderate" if score >= 0.33 else "low"
            results.append(RiskPrediction(
                condition=self.labels[index], horizon=f"next {self.horizon_years:g} years",
                risk_band=band, score_0_to_1=round(score, 8),
                evidence=[f"Delphi-compatible sequence of {len(encoded)} mapped events"],
                limitations=[
                    "Research-only generative estimate; requires a validated, calibrated checkpoint.",
                    "Unmapped events are excluded and age is approximated from the current reported age.",
                ],
            ))
        return results
