# Resources Catalog — Fixed Points of AI Revision

## Summary
Resources gathered for testing whether an iterative **review → revise** loop with
an AI converges to a fixed point. Papers (14 PDFs + 220-paper ranked pool),
two curated seed-document datasets, and a dependency-ready `.venv`. No external
code repos were required or specified.

## Papers
Total downloaded: **14** (search pool: 220 unique, ranked in `paper_search_results/merged.json`).

| Title | Authors | Year | File | Key info |
|-------|---------|------|------|----------|
| Do Language Models Converge to Themselves? Recursive Self-Refinement as Textual Relaxation | Wu et al. | 2026 | papers/converge_to_themselves.pdf | **Direct precedent, deep-read.** Defines exact/approx fixed points, exponential relaxation, all metrics. |
| Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | papers/self_refine.pdf | Canonical feedback→refine loop. |
| Reflexion | Shinn et al. | 2023 | papers/reflexion.pdf | Verbal self-feedback across attempts. |
| CRITIC | Gou et al. | 2023 | papers/critic_self_correct.pdf | Critique + external tools. |
| LLMs Cannot Self-Correct Reasoning Yet | Huang et al. | 2023 | papers/cannot_self_correct.pdf | Self-correction can degrade without oracle. |
| When Can LLMs Actually Correct Their Own Mistakes? (survey) | Kamoi et al. | 2024 | papers/critical_survey_when_correct_mistakes.pdf | Taxonomy of when it helps/hurts. |
| Pride and Prejudice: LLM Amplifies Self-Bias | Xu et al. | 2024 | papers/self_bias.pdf | Iterative refine amplifies self-bias → judge must differ. |
| Feedback Friction | — | 2025 | papers/feedback_friction.pdf | Models under-apply feedback. |
| Recursive Chain-of-Feedback | — | 2024 | papers/recursive_chain_feedback.pdf | Repeated feedback can degrade; mitigation. |
| Curse of Recursion | Shumailov et al. | 2023 | papers/curse_of_recursion.pdf | Model collapse (training). |
| Self-Consuming Generative Models Go MAD | Alemohammad et al. | 2023 | papers/self_consuming_mad.pdf | Self-consuming loops lose quality/diversity. |
| Self-Correction as Feedback Control | — | 2026 | papers/self_correction_feedback_control.pdf | Stability-threshold framing. |
| Experimental evidence of progressive ChatGPT self-convergence | Xylogiannopoulos et al. | 2026 | papers/progressive_chatgpt_selfconvergence.pdf | Independent self-convergence report. |
| Improving Iterative Text Revision by Learning Where to Edit | — | 2022 | papers/iterative_text_revision.pdf | Pre-LLM multi-step revision. |

See `papers/README.md` for grouping and `literature_review.md` for synthesis.

## Datasets
Total: **2 seed sets, 66 documents** (small JSON, kept in git).

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Scientific abstracts | arXiv API | 48 docs / 6 cats | Iterative revision seeds | datasets/documents/scientific_abstracts.json | Mirrors Wu et al. ICML setting. |
| General prose | Wikipedia API | 18 docs | Genre-generalization seeds | datasets/documents/general_prose.json | CC BY-SA. |

See `datasets/README.md` for schema, loading, and refresh instructions.

## Code Repositories
Total cloned: **0** (none specified/required). Reference implementations linked in
`code/README.md` (Self-Refine, Reflexion, CRITIC) for design patterns only.
Experiment dependencies installed in `.venv`: `anthropic`, `python-Levenshtein`,
`numpy`, `pypdf`, `requests`, `httpx`.

## Resource Gathering Notes

### Search strategy
3 diligent + 1 fast paper-finder sweeps: "iterative LLM self-refinement and
revision convergence to fixed point", "model collapse recursive training on
generated text", "LLM automated peer review feedback loop" (timed out → fast
retry "LLM as reviewer automated peer review"). 220 unique papers merged and
ranked by relevance (0–3) and citations.

### Selection criteria
The direct precedent (Wu et al.) + foundations (Self-Refine, Reflexion, CRITIC) +
the skeptical camp (cannot-self-correct, self-bias, feedback-friction) + the
dynamical-systems/collapse lens. Prioritized on-hypothesis papers over the large
model-collapse-training cluster (kept as interpretation only).

### Challenges encountered
- `find_papers.py` needed `httpx`; installed into `.venv`.
- One diligent sweep timed out; re-run in fast mode.
- arXiv "all:" title search mis-resolved a few titles (physics/unrelated papers);
  fixed via known arXiv IDs and per-PDF title verification with `pypdf`.
- 3 relevant 2026 papers are Semantic-Scholar-only (no arXiv PDF); abstracts
  retained in `paper_search_results/merged.json`.

### Gaps and workarounds
- No off-the-shelf dataset for "documents to iteratively revise" → built two
  curated seed sets (scientific + prose) from live APIs.
- No reference repo needed → experiment is a thin LLM-API harness; deps pre-installed.

## Recommendations for Experiment Design
1. **Primary dataset:** `scientific_abstracts.json` (48); add `general_prose.json`
   (18) for genre generalization. Subset 10–15/condition first to bound cost.
2. **Baselines/conditions:** single-stage self-refine vs two-stage review→revise;
   temp 0 vs default; same-model vs cross-model reviewer; two genres; a review-framing variant.
3. **Metrics:** normalized Levenshtein `Δt`; exact/approx fixed points
   (ε_edit=0.02, ε_wc=0.03, 2 consecutive); exponential fit `A e^{-kt}+c` (`R²`);
   semantic-drift cosine; LLM-judge (different model) for clarity/fidelity/meaning.
4. **Code to reuse:** none to clone; follow prompt/stopping patterns from Self-Refine.
5. **Cautions:** judge ≠ reviser (self-bias); log all `V_t`/`R_t`; add an
   identity/paraphrase control for the `Δt` noise floor; distinguish genuine
   stabilization from premature convergence that ignores the review.
