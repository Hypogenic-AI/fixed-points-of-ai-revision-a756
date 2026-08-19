# Planning & Direction Budget — Fixed Points of AI Revision

## Hypothesis under test
> If we iteratively feed a paper to an AI for **review**, then provide both the
> review and the paper to the AI to make **revisions**, and repeat this process,
> will the process converge to a **fixed point** where further iterations produce
> no changes?

**Formalization.** Let a document version be `V_t`. The loop is a *two-stage*
operator applied by the same model family:

```
R_t   = Review(V_t)                # stage 1: AI critiques the current version
V_{t+1} = Revise(V_t, R_t)         # stage 2: AI rewrites given (version + review)
```

This defines a discrete dynamical system `V_{t+1} = F(V_t)` where
`F = Revise(·, Review(·))`. We ask whether the trajectory `V_0, V_1, …` reaches
a **fixed point** (`V_{t+1} = V_t`, *exact*) or a **soft fixed-point region**
(consecutive versions differ by only a tiny, non-decreasing edit distance,
*approximate*).

**Key distinction from the closest prior work.** Wu et al. 2026 ("Do Language
Models Converge to Themselves?") study a *single-stage* self-refine operator
(revise directly). Our hypothesis specifies a **decoupled critique→revise loop**.
Whether inserting an explicit, freshly-generated review at every step changes the
convergence behavior (faster/slower, more/less stable, quality up or down) is the
distinctive, under-studied question this project targets.

---

## Enumerated research directions (scored, then pruned)

Scoring rubric (1–5 each): **Ev**=literature evidence/grounding,
**Rel**=relevance to hypothesis, **IG**=expected information gain,
**Feas**=implementation feasibility for an automated LLM experiment.

| # | Direction | Ev | Rel | IG | Feas | Total | Decision |
|---|-----------|----|----|----|------|-------|----------|
| D1 | **Convergence dynamics of the review→revise loop** — does the two-stage loop reach exact/approximate fixed points; measure convergence time, exponential-relaxation fit `Δt ≈ A e^{-kt}+c`, residual floor. | 5 | 5 | 5 | 5 | **20** | **KEEP** |
| D2 | **Determinants of convergence** — ablate decoding temperature (0 vs default), reviewer=reviser vs cross-model, document genre (scientific abstract vs general prose), review framing (harsh/lenient/structured), and single-stage self-refine vs two-stage review→revise. | 5 | 5 | 5 | 4 | **19** | **KEEP** |
| D3 | **Quality trajectory & failure modes** — does convergence coincide with genuine improvement (LLM-as-judge on clarity/fidelity/meaning) or with stagnation, self-bias amplification, semantic drift, or collapse-like degradation? | 5 | 4 | 5 | 4 | **18** | **KEEP** |
| D4 | Fixed-point uniqueness / basins of attraction — do different initial drafts of the same content converge to one attractor or is it path-dependent? | 3 | 4 | 4 | 3 | 14 | PRUNE → folded into D1 as an optional sub-analysis (re-seed a few docs with paraphrased `V_0`). |
| D5 | Formal contraction-mapping / Markov-chain theory of convergence conditions. | 3 | 3 | 3 | 2 | 11 | PRUNE → keep only a light dynamical-systems framing in the writeup; full theory is future work (harder to validate empirically at this budget). |
| D6 | Recursive **fine-tuning** on the model's own revisions (model-collapse angle). | 4 | 2 | 4 | 1 | 11 | PRUNE → out of scope: hypothesis is *inference-time* revision, not training; fine-tuning cost is prohibitive. Kept only as an interpretive lens (Shumailov, "Curse of Recursion"). |

### Top-3 kept for implementation
1. **D1 — Convergence dynamics of the review→revise loop** (the direct test).
2. **D2 — Determinants of convergence** (what controls whether/when it converges).
3. **D3 — Quality trajectory & failure modes** (is the fixed point good or degenerate?).

### Pruning record
- D4 folded into D1 (cheap add-on, not a separate track).
- D5 reduced to framing only; no separate theory experiment.
- D6 dropped (training out of scope, cost); retained only as interpretation.

Per the direction budget, the search space will **not** be expanded later unless
new evidence invalidates this ranking; any change must be justified in `STATE.md`.

---

## Concrete experimental plan (for experiment_runner)

**Operator.** Two prompts: a `Review` prompt (produce a critique of the current
version) and a `Revise` prompt (rewrite the version to address the review while
preserving all claims/meaning). Same model family for both by default (Claude,
`claude-sonnet-5` recommended as the workhorse; see `code/README.md`).

**Trajectories.** For each seed document `V_0`, run `T = 8–10` iterations,
logging every `V_t` and `R_t`.

**Metrics (mirroring and extending Wu et al. 2026):**
- Normalized Levenshtein edit distance `Δt = lev(V_t, V_{t+1}) / max(|V_t|,|V_{t+1}|)`.
- Exact fixed point (`V_{t+1}=V_t`) and approximate fixed point
  (`Δt < ε_edit=0.02` and relative word-count change `< ε_wc=0.03`, for 2
  consecutive steps).
- First-exact / first-approx convergence time; final-step `Δt`; zero-final rate.
- Exponential-relaxation fit `Δt ≈ A e^{-kt} + c` (report `R²`, `k`, floor `c`).
- Semantic drift: cosine similarity of `V_0` vs `V_t` embeddings (meaning preservation).
- LLM-as-judge (separate model, e.g. a different family): clarity, conciseness,
  technical fidelity, meaning preservation, best-version pick over `V_0, V_1, V_T`.

**Conditions (D2):** temperature ∈ {0, default}; genre ∈ {scientific, prose};
reviewer=reviser vs cross-model; review framing ∈ {structured, harsh, lenient};
single-stage vs two-stage.

**Primary datasets:** `datasets/documents/scientific_abstracts.json` (48 arXiv
abstracts, 6 fields) and `datasets/documents/general_prose.json` (18 Wikipedia
intros). Start with a 10–15 doc subset per condition to bound API cost, then scale.

**Success criteria.** The hypothesis is supported if trajectories reach exact or
approximate fixed points for a large majority of documents, with `Δt` decaying
monotonically (well-fit by exponential relaxation). It is refuted/qualified if
trajectories oscillate, drift indefinitely, or degrade in quality without stabilizing.
