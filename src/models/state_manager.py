from src.models.schemas import PatientState, StatePatch, Symptom


LIST_FIELDS = (
    "associated_symptoms",
    "recent_events_or_exposures",
    "relevant_negatives",
)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _merge_symptoms(old: list[Symptom], new: list[Symptom]) -> list[Symptom]:
    by_name = {symptom.name.casefold(): symptom.model_copy(deep=True) for symptom in old}
    for incoming in new:
        key = incoming.name.casefold()
        if key not in by_name:
            by_name[key] = incoming
            continue
        current = by_name[key]
        update = incoming.model_dump(exclude_none=True)
        for list_field in ("aggravating_factors", "relieving_factors"):
            if list_field in update:
                update[list_field] = _unique(
                    getattr(current, list_field) + update[list_field]
                )
        by_name[key] = current.model_copy(update=update)
    return list(by_name.values())


def apply_patch(state: PatientState, patch: StatePatch) -> PatientState:
    updated = state.model_copy(deep=True)
    for field in ("chief_complaint", "age", "sex_at_birth", "pregnancy_possible"):
        value = getattr(patch, field)
        if value is not None:
            setattr(updated, field, value)
    updated.symptoms = _merge_symptoms(updated.symptoms, patch.symptoms)
    for field in LIST_FIELDS:
        incoming = getattr(patch, field)
        if incoming:
            existing = getattr(updated, field)
            setattr(updated, field, _unique(existing + incoming))
    return updated
