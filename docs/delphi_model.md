# CPU Delphi demonstration model

## Reference architecture

The implementation follows Shmatko et al., *Learning the natural history of human disease
with generative transformers*, Nature (2025), DOI
[`10.1038/s41586-025-09529-3`](https://doi.org/10.1038/s41586-025-09529-3), and the authors'
[`gerstung-lab/delphi`](https://github.com/gerstung-lab/delphi) code release.

The paper model represents diagnoses, sex and selected lifestyle variables as tokens; replaces
ordinary position embeddings with continuous age encodings; and learns competing disease-event
rates and time to the next event. It was developed from approximately 400,000 UK Biobank
participants and externally evaluated on approximately 1.9 million Danish registry records.
The full UK Biobank training data require approved researcher access. The trained weights are
distributed separately by the authors under CC BY-NC-ND 4.0 and are not included here.

## What this repository includes

`src/delphi/model.py` is a CPU-capable implementation of the essential architecture. Its default
configuration mirrors the compact published shape (12 layers, 12 heads, embedding width 120,
sequence length 96). `scripts/train_delphi.py` trains a compatible checkpoint and
`src/delphi/predictor.py` loads it for next-event horizon scoring.

The five bundled JSONL trajectories are **software fixtures**, not observations and not a
replacement for the UK Biobank dataset. Their disease/risk-factor ordering is derived from the
WHO CVD and COPD fact sheets, CDC diabetes risk-factor material, and NICE NG222. Artificial ages
exist only to exercise age encoding and time-loss code. These fixtures are small because they
test the data contract and CPU execution; they cannot estimate incidence, calibrate risk or
validate a model. Training them produces a demonstration checkpoint whose output must never be
interpreted as a medical probability.

## Runtime routing

The application considers Delphi available only when a checkpoint and its exact vocabulary are
present and compatible. With a usable checkpoint, predicted next-event labels are added to the
RAG query and passed with their scores to the verdict LLM. Without one, the application follows
the RAG-only path and produces no substitute score.

## Requirements for a real model

A research implementation needs appropriately governed longitudinal event data, the original
token/ICD mapping, population and outcome definitions, censoring logic, training/validation
splits, calibration, external validation, subgroup analysis and prospective governance. Do not
use the demonstration checkpoint or this application for clinical decisions.
