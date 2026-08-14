from src.models.schemas import HistoryPatch, PatientHistory


LIST_FIELDS = (
    "conditions", "previous_symptoms", "medication_history", "allergies",
    "procedures", "family_history", "lifestyle_factors",
)


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    result, seen = [], set()
    for value in left + right:
        key = value.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def apply_history_patch(history: PatientHistory, patch: HistoryPatch) -> PatientHistory:
    updated = history.model_copy(deep=True)
    for field in LIST_FIELDS:
        setattr(updated, field, _merge_unique(getattr(updated, field), getattr(patch, field)))
    known_ids = {event.id for event in updated.events}
    updated.events.extend(event for event in patch.events if event.id not in known_ids)
    updated.explicitly_no_history = updated.explicitly_no_history or patch.explicitly_no_history
    updated.collection_complete = updated.explicitly_no_history or any(
        getattr(updated, field) for field in LIST_FIELDS
    ) or bool(updated.events)
    return updated
