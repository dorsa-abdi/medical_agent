from datetime import date

from src.models.schemas import HealthTimeline, PatientHistory, PatientState, TimelineEvent


class TimelineBuilder:
    def build(self, state: PatientState, history: PatientHistory) -> HealthTimeline:
        today = date.today()
        events = list(history.events)
        dated_labels = {(event.type, event.label.casefold()) for event in events}

        def add_undated(event_type, labels, status="historical"):
            for label in labels:
                if label.casefold() in {"none", "none reported", "no relevant history"}:
                    continue
                key = (event_type, label.casefold())
                if key not in dated_labels:
                    events.append(TimelineEvent(
                        date=today, type=event_type, label=label, status=status,
                        source="patient_reported_undated", metadata={"date_precision": "unknown"},
                    ))
                    dated_labels.add(key)

        add_undated("diagnosis", history.conditions)
        add_undated("medication", history.medication_history)
        add_undated("procedure", history.procedures)
        add_undated("lifestyle", history.lifestyle_factors)
        for symptom in state.symptoms:
            events.append(TimelineEvent(
                date=today, type="symptom", label=symptom.name, value=symptom.severity_0_to_10,
                status="active", metadata={"onset": symptom.onset, "duration": symptom.duration},
            ))
        ordered = sorted(events, key=lambda event: event.date)
        quality = "adequate" if len(ordered) >= 5 else "limited" if len(ordered) >= 2 else "insufficient"
        return HealthTimeline(patient_id=state.patient_id, events=ordered, data_quality=quality)
