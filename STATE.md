# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-08-20T01:09:23.133833Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/fixed-points-of-ai-revision-a756`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder — COMPLETE.

**Completed:**
- Isolated `.venv` (uv) with `pyproject.toml` (no parent pollution). Deps:
  pypdf, requests, httpx, anthropic, python-Levenshtein, numpy.
- 4 paper-finder sweeps → 220 unique ranked papers (`paper_search_results/merged.json`).
- 14 PDFs in `papers/`. Deep-read the direct precedent
  `converge_to_themselves.pdf` (Wu et al. 2026, arXiv:2607.22653).
- 2 curated seed datasets (66 docs): `datasets/documents/scientific_abstracts.json`
  (48 arXiv abstracts, 6 cats) + `general_prose.json` (18 Wikipedia intros).
- Deliverables written: `literature_review.md`, `resources.md`, `planning.md`,
  `papers/README.md`, `datasets/README.md`, `code/README.md`, `datasets/.gitignore`.

**Key findings / decisions:**
- Direct precedent Wu et al. 2026 shows *single-stage* self-refinement relaxes to a
  soft fixed point (temp 0: 100% exact FP, mean 2.6 iters; exp. fit R²=0.990).
  Metrics adopted: normalized Levenshtein Δt; exact FP (V_{t+1}=V_t); approx FP
  (Δt<ε_edit=0.02 AND rel. wc change<ε_wc=0.03, 2 consecutive); fit Δt≈A e^{-kt}+c.
- **Our novelty = decoupled two-stage review→revise loop** (precedent fuses/omits
  the explicit review). Direction budget → top 3: D1 convergence dynamics of the
  review→revise loop; D2 determinants (temp, cross-model reviewer, genre, framing,
  single- vs two-stage); D3 quality-vs-degeneracy at the fixed point.
  Pruned: D4 basins (fold into D1), D5 formal theory (framing only), D6 recursive
  fine-tuning/model-collapse (out of scope, cost). See `planning.md`.
- Caution from literature: judge model must differ from reviser (self-bias, Xu 2024);
  watch premature convergence that ignores the review (Feedback Friction).

**Next phase (experiment_runner) — concrete steps:**
1. Build `code/run_loop.py`: `Review(V_t)`→R_t then `Revise(V_t,R_t)`→V_{t+1},
   T≈8–10, log all V_t/R_t. Reviser `claude-sonnet-5`; judge a different model.
2. Compute metrics in `planning.md`; fit exponential relaxation (numpy).
3. Run D1 on ~10–15 docs/condition, then D2 ablations (temp 0/default, cross-model,
   genre, single- vs two-stage), then D3 (LLM-judge + semantic-drift cosine).
4. Requires `ANTHROPIC_API_KEY`; fail loudly if missing (don't mock silently).

**Uncertainties:** 3 relevant 2026 papers are Semantic-Scholar-only (abstracts in
merged.json, no PDF). Workspace note: pipeline_state recorded actual cwd `/app`
but this phase ran and wrote correctly under `/workspaces/fixed-points-of-ai-revision-a756`.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner — IN PROGRESS.

**Key deviation from resource_finder plan (justified):** Only `OPENAI_API_KEY`
is present (no OPENROUTER/ANTHROPIC). So the reviser/workhorse is `gpt-4.1` and
the independent cross-model reviewer + LLM-judge is `gpt-4o` (a different model
generation) — this still honours the "judge != reviser" self-bias caution
across genuine model generations. Embeddings: `text-embedding-3-small`.

**Harness (`code/`):** `llm.py` (disk-cached OpenAI calls, truncation +
token/cost tracking), `prompts.py`, `run_loop.py` (two_stage & single_stage),
`metrics.py` (norm-Levenshtein Δt, exact/approx FP, exp-relaxation fit, cosine),
`experiments.py` (8 conditions, parallel over docs), `analyze.py` (drift, LLM
judge, stats, figures).

**Early empirical finding (pre-full-run):** the decoupled two-stage
review→revise loop does NOT reach a fixed point in 8 iters on a scientific
abstract even with length-preserving instructions — Δt plateaus at a high floor
(exp-fit c≈0.34, R²≈0.73) and the document keeps being substantially rewritten
and grows in length. This is the opposite of Wu et al.'s single-stage relaxation
and is the core novel result to confirm across conditions.

**Added ablation:** `D2_unconstrained` (no length guidance) to isolate the
length-inflation phenomenon vs the length-preserving default operator.

**Next steps:** finish full run (all 8 conditions), run `analyze.py`
(judge=gpt-4o), then write REPORT.md/README.md.

---
**Phase: experiment_runner — COMPLETE (2026-08-20).**

All 8 conditions ran to completion (12 docs each, 10 for prose; V_0..V_8).
`analyze.py` run to completion (fixed matplotlib `labels`->`tick_labels`; all
LLM/embed calls cached, re-run cost $0). Artifacts on disk: `results/summary.json`,
`results/stats.json`, `results/per_trajectory.{json,csv}`, 6 figures in `figures/`,
`REPORT.md`, `README.md`.

**Headline result (hypothesis answered = NO for two-stage loop):**
- Two-stage review->revise does NOT converge: D1 mean final Δ=0.376, 0% exact/approx
  FP, exp-fit floor c=0.31; t(11)=11.3 vs 0, p=2e-7. Robust to truncation control
  (clean min Δ=0.34 >> 0.02, p=5e-8).
- Single-stage self-refine DOES converge (final Δ=0.048, 42% approx FP, R²=0.98) —
  replicates Wu et al. 2026. Single-vs-two-stage is the dominant effect
  (Cohen dz=-3.26, p=5e-4). The explicit review stage causes non-convergence.
- Mechanism: runaway length inflation (~7x chars; prose 15x) because the reviewer is
  never satisfied; meaning preservation 10->6.2/10, cos(V0,Vfin) 0.97->0.77, yet
  isolated quality judge flat 7.5->8.0 (self-bias signature). Length-preserving
  instruction largely ignored (D1 vs unconstrained n.s., p=0.62).
- Determinants: temp default worse/chaotic (dz=+2.09, R²=0.27); cross-model & lenient
  reviewer roughly halve churn but still no FP; harsh n.s.

**Known limitations documented in REPORT.md:** single provider (OpenAI, only key
present), 2048-token cap causes late-step truncation (mitigated by clean-Δ metric),
T=8 horizon, n=12/cond, LLM-judged fidelity. Session complete.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
