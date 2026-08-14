# Agent architecture

## Intake

1. The emergency screen runs before every model call.
2. The first safe message is extracted into `PatientState`, which contains the current encounter
   only.
3. The assistant immediately requests prior conditions, medications, allergies, procedures,
   family history, lifestyle and dates.
4. The response is extracted into the separate `PatientHistory`. An explicit no-history response
   is represented rather than inferred from empty lists.
5. Current-intake validation and focused follow-up continue without mixing historical facts into
   the current encounter.

## Analysis router

`PatientState` and `PatientHistory` create a chronological timeline. When a compatible checkpoint,
vocabulary, age and mapped events are present, the CPU Delphi adapter produces next-event horizon
scores. Predicted labels are appended to the retrieval query. Otherwise the router uses RAG only
and emits no numerical score.

The four retrieval contexts are clinical safety, disease knowledge, dynamic patient memory and
decision context. The verdict agent receives the separate state/history objects, timeline,
optional Delphi output and retrieved chunks. It returns a validated report with interpretation,
precautions, questions, warning signs, uncertainty and citations. Invalid LLM output falls back
to a deterministic evidence summary.
