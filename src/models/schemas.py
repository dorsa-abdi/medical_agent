from datetime import UTC, date, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Symptom(BaseModel):
    name: str
    onset: str | None = None
    duration: str | None = None
    severity_0_to_10: int | None = Field(default=None, ge=0, le=10)
    location: str | None = None
    quality: str | None = None
    pattern: str | None = None
    aggravating_factors: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)


class PatientState(BaseModel):
    schema_version: str = "2.0"
    patient_id: str = Field(default_factory=lambda: str(uuid4()))
    chief_complaint: str | None = None
    symptoms: list[Symptom] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    age: int | None = Field(default=None, ge=0, le=125)
    sex_at_birth: Literal["female", "male", "intersex", "unknown"] | None = None
    pregnancy_possible: bool | None = None
    recent_events_or_exposures: list[str] = Field(default_factory=list)
    relevant_negatives: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    intake_complete: bool = False
    turn_count: int = 0
    pending_field: str | None = None
    pending_attempts: int = Field(default=0, ge=0)
    unavailable_fields: list[str] = Field(default_factory=list)
    safety_level: Literal["routine", "urgent", "emergency"] | None = None
    safety_reason: str | None = None
    safety_requires_escalation: bool = False


class StatePatch(BaseModel):
    chief_complaint: str | None = None
    symptoms: list[Symptom] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    age: int | None = Field(default=None, ge=0, le=125)
    sex_at_birth: Literal["female", "male", "intersex", "unknown"] | None = None
    pregnancy_possible: bool | None = None
    recent_events_or_exposures: list[str] = Field(default_factory=list)
    relevant_negatives: list[str] = Field(default_factory=list)


class PatientHistory(BaseModel):
    """Longitudinal facts kept separate from the current encounter state."""
    schema_version: str = "1.0"
    conditions: list[str] = Field(default_factory=list)
    previous_symptoms: list[str] = Field(default_factory=list)
    medication_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    lifestyle_factors: list[str] = Field(default_factory=list)
    events: list["TimelineEvent"] = Field(default_factory=list)
    explicitly_no_history: bool = False
    collection_complete: bool = False


class HistoryPatch(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    previous_symptoms: list[str] = Field(default_factory=list)
    medication_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    lifestyle_factors: list[str] = Field(default_factory=list)
    events: list["TimelineEvent"] = Field(default_factory=list)
    explicitly_no_history: bool = False


class ValidationResult(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_sufficient: bool = False
    next_field: str | None = None


class TurnResult(BaseModel):
    state: PatientState
    history: PatientHistory
    reply: str
    status: Literal["collecting", "complete", "urgent"]


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TimelineEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    date: date
    type: Literal["symptom", "diagnosis", "medication", "test", "procedure", "visit", "lifestyle", "other"]
    label: str
    value: str | float | int | None = None
    unit: str | None = None
    status: Literal["active", "resolved", "historical", "unknown"] = "unknown"
    source: str = "patient_reported"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthTimeline(BaseModel):
    patient_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    events: list[TimelineEvent] = Field(default_factory=list)
    data_quality: Literal["insufficient", "limited", "adequate"] = "insufficient"


class RiskPrediction(BaseModel):
    condition: str
    horizon: str = "next 1-5 years"
    risk_band: Literal["low", "moderate", "high", "unknown"]
    score_0_to_1: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    corpus: Literal["clinical", "disease", "patient_memory", "decision"]
    document_id: str
    title: str
    text: str
    source: str
    score: float = Field(ge=0, le=1)


class ClinicalReport(BaseModel):
    summary: str
    risk_interpretation: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    questions_for_clinician: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)
    uncertainty: str
    citations: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Research prototype only. This output is not a diagnosis, treatment plan, "
        "or substitute for evaluation by a qualified clinician."
    )


class AnalysisResult(BaseModel):
    timeline: HealthTimeline
    predictions: list[RiskPrediction]
    retrieved_context: list[RetrievedChunk]
    report: ClinicalReport
    used_longitudinal_model: bool
    risk_backend: Literal["delphi", "rag_only", "none"] = "none"
    model_notice: str
