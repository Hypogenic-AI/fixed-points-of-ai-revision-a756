# Datasets — Seed Documents for Iterative Revision

These are the **initial documents `V_0`** that the review→revise loop iterates on.
They are small, curated JSON seed files (kept in git for reproducibility). No large
binary data is stored here.

## Dataset 1: Scientific Abstracts (primary)
- **File:** `documents/scientific_abstracts.json`
- **Source:** arXiv API (`export.arxiv.org`), most-recent submissions per category.
- **Size:** 48 documents (54–291 words each; mean ~174).
- **Categories (6):** `cs.AI`, `cs.CL`, `q-bio.NC`, `econ.GN`, `math.OC`, `stat.ML`.
- **Genre:** `scientific_abstract` — mirrors the ICML-abstract setting of Wu et al. 2026.
- **Why:** Scientific abstracts are the established testbed; style edits must preserve
  technical claims/numbers, making meaning-preservation measurable.

## Dataset 2: General Prose (generalization test)
- **File:** `documents/general_prose.json`
- **Source:** Wikipedia REST summary API (CC BY-SA).
- **Size:** 18 documents (encyclopedia introductions, ~50–290 words).
- **Genre:** `general_prose` — tests whether convergence dynamics are genre-specific.

## Combined & samples
- `documents/all_documents.json` — all 66 docs merged.
- `documents/samples.json` — 5-record preview.

## Schema
```json
{
  "doc_id": "abs_000",
  "title": "…",
  "text": "… the document body that gets iteratively revised …",
  "category": "cs.AI",
  "genre": "scientific_abstract"
}
```

## Loading
```python
import json
docs = json.load(open("datasets/documents/all_documents.json"))
for d in docs:
    v0 = d["text"]           # initial version fed into the review->revise loop
```

## Reproducing / refreshing the seeds
Scientific abstracts (arXiv API):
```bash
python /tmp/build_ds.py          # script text is embedded in STATE.md notes / git history
```
General prose (Wikipedia summaries): fetch
`https://en.wikipedia.org/api/rest_v1/page/summary/<Title>` and keep the `extract`.
Both builders live in the resource_finder run log; regenerating pulls fresh
newest-first items, so exact documents will differ but the schema is stable.

## Notes
- Keep the **title** fixed and revise only **text** (as in Wu et al.), so titles
  anchor the topic while the body relaxes.
- Start experiments on a 10–15 doc subset per condition to bound LLM API cost,
  then scale to the full set.
- License: arXiv abstracts are used for research; Wikipedia text is CC BY-SA.
