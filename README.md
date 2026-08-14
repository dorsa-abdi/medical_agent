# Medical history, Delphi and RAG prototype

Research-only Streamlit application with a safety-first intake, separate current and historical
patient models, optional CPU Delphi-compatible next-event scoring, four local RAG indexes and a
grounded verdict LLM. It does not diagnose, prescribe or replace professional evaluation.

## Runtime flow

1. Screen every patient message for deterministic emergency phrases.
2. If the initial complaint is not urgent, preserve it as current `PatientState` and ask for
   previous/ongoing diseases, medications, allergies, procedures, family history, lifestyle and
   approximate dates. Store these only in `PatientHistory`.
3. Build a typed timeline from `PatientHistory` plus current symptoms.
4. Route analysis:
   - usable Delphi checkpoint: obtain next-disease horizon scores, add predicted disease labels
     to the RAG query, and pass scores plus evidence to the verdict LLM;
   - no checkpoint: run all RAGs and the verdict LLM without generating any risk score.
5. Return a cited summary, uncertainty, precautions, warning signs and discussion points.

The four stores are clinical safety, disease knowledge, dynamic patient memory and clinical
decision context. TF-IDF keeps retrieval local, inspectable and lightweight.

## Install and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                   # Windows: copy .env.example .env
streamlit run app.py
```

The first LLM call downloads `Qwen/Qwen2.5-0.5B-Instruct`. Generation is lazy and runs on CPU by
default. Set `USE_LLM_REASONING=false` for the deterministic verdict fallback.

## Google Colab

Download `medical_agent_colab.ipynb` and the Colab project ZIP. Open the notebook in Colab,
run the cells in order and upload the ZIP when prompted. Streamlit is not installed or used in
the notebook. The conversation uses notebook `input()` and all assistant questions and structured
extraction are produced through `Qwen/Qwen2.5-0.5B-Instruct`. The notebook detects a Colab GPU for
Qwen and automatically trains the small Delphi demonstration checkpoint before the conversation.

The reusable notebook interface is implemented in `src/interfaces/colab_chat.py`. A fresh session
can also be started directly from a notebook cell:

```python
from src.interfaces.colab_chat import ColabMedicalSession, run_interactive_session

session = ColabMedicalSession(device="cuda", train_demo_delphi=True, debug=True)
run_interactive_session(session)
```

With `debug=True`, every turn prints the raw conversation memory, the current `PatientState`
and the separate `PatientHistory`. Set it to `False` after debugging; these views can contain
sensitive health text and should never be enabled in production logs.

The most recent eight messages, the complete structured current state, the separate history and
the field requested by the previous question are supplied to Qwen on every intake turn. The raw
conversation supports short contextual answers; the typed objects remain the authoritative
memory used by timeline construction and analysis.

After model extraction, the validator independently checks the raw response against the exact
previously requested field. It accepts narrow forms for age, onset, 0–10 severity and associated
symptoms, then advances to the next missing field. If it cannot map an answer, the assistant says
what was not recorded and shows the expected format. Questions marked
`[Fallback question — ...]` came from the deterministic fallback and include the exact reason:
empty response, timeout or exception. `ValidatorAgent` alone selects `next_field`; the Questioner
receives only that field and uses Qwen to phrase one question. Any non-empty response returned
within `QUESTION_TIMEOUT_SECONDS` is accepted, including Persian `؟` and responses without final
punctuation.

## CPU Delphi demonstration

Train the small architecture on the bundled source-derived software fixtures:

```bash
python scripts/train_delphi.py \
  --data data/delphi/demo_trajectories.jsonl \
  --vocabulary data/delphi/vocabulary.json \
  --output checkpoints/delphi.pt \
  --epochs 3
```

Configure `.env`:

```env
DELPHI_CHECKPOINT=checkpoints/delphi.pt
DELPHI_VOCABULARY=data/delphi/vocabulary.json
DELPHI_HORIZON_YEARS=5
```

The demo checkpoint proves CPU training/inference and routing only. Its output is not a medical
probability. See [`docs/delphi_model.md`](docs/delphi_model.md) for the Nature paper, official
code, original data scale, weight availability and limitations.

## Bundled knowledge

The compact RAG corpus contains paraphrased chunks derived from WHO cardiovascular/COPD sources,
CDC stroke/diabetes sources, NICE shared-decision/depression guidance and HL7 FHIR. Each chunk
retains its URL. [`docs/knowledge_sources.md`](docs/knowledge_sources.md) explains the complete
sources, included subset, selection rationale and update requirements.

## Input examples

`examples/timeline.json` demonstrates the optional timeline import format. Direct identifiers
must not be entered. Production handling of health records requires consent, encryption, access
controls, retention rules, audit trails, jurisdiction-specific compliance and clinical governance.

## Important limitations

- The emergency phrase screen is not a validated triage system.
- A Delphi architecture without appropriately governed training data and validated weights has
  no clinical predictive value.
- RAG retrieval does not ensure that a passage applies to an individual.
- The verdict LLM is constrained and validated structurally, but still requires professional
  review and may omit important context.
- Never start, stop or change medication based on this application.
