# Knowledge corpus provenance

The bundled corpus is intentionally small so the CPU-only prototype starts quickly. It is
not a substitute for the complete source documents and is not a clinical guideline database.
Every chunk is a concise paraphrase written for retrieval, retains a stable local identifier,
and links to its authoritative source.

| Source | Full source | Included subset | Why included |
|---|---|---|---|
| WHO cardiovascular diseases fact sheet | Global overview of CVD types, burden, risk factors, warning signs and prevention | Major modifiable/intermediate risk factors and urgent heart attack/stroke signs | Supports emergency context, CVD retrieval and prevention discussions |
| CDC stroke signs and symptoms | Public-health description of major stroke warning signs and emergency response | Five principal sudden-onset warning-sign groups | Independent authoritative support for safety-oriented retrieval |
| CDC diabetes risk factors and prevention | Type 1, type 2 and gestational diabetes risk information and prevention resources | Major type 2 risk factors and clinician-discussion context | Supports diabetes-related predicted-event queries without prescribing care |
| WHO COPD fact sheet | COPD causes, symptoms, impact, diagnosis, management and prevention | Common symptoms and major exposure risks | Supports respiratory retrieval for cough, breathlessness, smoking and pollution |
| NICE NG222 | Full evidence-based adult depression treatment and management guideline | High-level assessment factors only | Supports mental-health context while avoiding treatment generation |
| NICE NG197 | Full shared decision-making recommendations | Collaborative decision principle | Constrains next steps to clinician–patient discussion rather than autonomous advice |
| HL7 FHIR R5 | Complete healthcare data-exchange specification | Resource separation, provenance and observation concepts | Informs the separate history/timeline representation, not clinical conclusions |

## Updating the corpus

The source pages are broader and change over time. Before evaluation or deployment, a clinical
information specialist should review the current full documents, approve exact chunks, record
publication/review dates and jurisdiction, and run retrieval-quality and citation-grounding
evaluation. Do not silently replace a source while retaining its document ID.
