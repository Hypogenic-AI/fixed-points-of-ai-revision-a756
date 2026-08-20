# Fixed Points of AI Revision — Research Report

## 1. Executive Summary

**Research question.** If we iteratively feed a document to an AI for *review*,
then feed both the review and the document back to the AI to *revise*, and repeat,
does the process converge to a **fixed point** where further iterations produce no
changes?

**Key finding.** No. The decoupled two‑stage **review → revise** loop does **not**
converge to a fixed point. Across 12 scientific abstracts and 8 iterations, the
per‑step change plateaus at a large residual floor (mean final normalized edit
distance **Δ ≈ 0.38**, vs. the approximate‑fixed‑point threshold of 0.02;
one‑sample *t* = 11.3, *p* = 2×10⁻⁷), and **0%** of trajectories reach an exact or
approximate fixed point. The critical driver is the explicit review stage: a
single‑stage self‑refinement baseline run on the *same* documents **does** converge
(mean final Δ ≈ 0.05, 42% reach an approximate fixed point; exponential‑relaxation
fit R² = 0.98), replicating the "textual relaxation" result of Wu et al. (2026).
Inserting a freshly generated peer review at every step turns a convergent
relaxation into a **non‑converging expansion process**.

**Why it happens (mechanism).** A peer reviewer, unlike a "make this better"
editor, is *never satisfied*: it always emits new, specific demands ("define this
term," "add an example," "expand the discussion"). The reviser satisfies these
mostly by **adding text**, so documents inflate **~7×** in length over 8 iterations
(a 1.5 KB abstract becomes a ~10 KB mini‑paper), drift semantically away from the
original (cosine V₀↔V_final falls from 0.97 in the single‑stage baseline to 0.77),
and lose factual fidelity (LLM‑judged meaning preservation drops from 10/10 to
6.2/10, with ~4 added claims per document). Meanwhile the isolated‑quality judge
score barely moves (7.5 → 8.0), so the loop *feels* like it is improving while it
is actually diverging — exactly the self‑bias / feedback‑friction failure mode the
skeptical literature predicts.

**Practical implication.** "Keep asking an AI to review and revise until it
stabilizes" is not a reliable stopping strategy: the review stage supplies an
endless stream of edit pressure and the loop degrades meaning rather than settling.
If a fixed point is desired, either drop the separate review stage (single‑stage
self‑refine converges) or add an explicit convergence/stop criterion.

---

## 2. Research Question & Motivation

We test the hypothesis that the discrete dynamical system

```
R_t     = Review(V_t)            # stage 1: AI critiques the current version
V_{t+1} = Revise(V_t, R_t)       # stage 2: AI rewrites given (version + review)
```

i.e. `V_{t+1} = F(V_t)` with `F = Revise(·, Review(·))`, reaches a fixed point
`V_{t+1} = V_t` (exact) or a soft fixed‑point region (consecutive versions differ
by a tiny, stable edit distance; approximate).

**Why it matters.** Iterative "AI reviews then revises" loops are increasingly
used in practice (automated editing assistants, agentic writing pipelines,
self‑improving document workflows). Whether such loops *terminate* — and whether
their limit is a genuinely better document or a degenerate attractor — determines
whether they are safe to run to "stability."

**Gap in existing work.** The closest precedent, Wu et al. (2026, *Do Language
Models Converge to Themselves?*), studies a **single‑stage** self‑refine operator
(revise directly) and finds it relaxes to a soft fixed point. The canonical
Self‑Refine loop (Madaan et al., 2023) *fuses* critique and revision in one prompt.
Neither isolates the effect of a **decoupled, freshly regenerated review at every
step** on the convergence dynamics. The skeptical literature (Huang et al. 2023;
Xu et al. 2024, self‑bias; Feedback Friction 2025; model‑collapse work) predicts
recursion need not be benign, but does not measure the review→revise fixed‑point
question directly. This project fills that gap.

---

## 3. Experimental Setup

### Operator and conditions

Two prompts implement the operator (full text in `code/prompts.py`): a **Review**
prompt (produce a 5–8 bullet critique; do **not** rewrite) and a **Revise** prompt
(rewrite to address the review, preserve meaning/claims/scope, "keep roughly the
same length"). For each seed document we run **T = 8** iterations, logging every
`V_t` and `R_t`.

We test **8 conditions** (a direct test D1 + seven D2 ablations):

| Condition | Mode | Temp | Reviewer | Framing | Genre | Length guidance |
|---|---|---|---|---|---|---|
| **D1_main** | two‑stage | 0.0 | gpt‑4.1 (self) | structured | scientific | preserve |
| D2_single_stage | **single‑stage** | 0.0 | — | — | scientific | preserve |
| D2_temp_default | two‑stage | **1.0** | gpt‑4.1 | structured | scientific | preserve |
| D2_cross_model | two‑stage | 0.0 | **gpt‑4o** | structured | scientific | preserve |
| D2_framing_harsh | two‑stage | 0.0 | gpt‑4.1 | **harsh** | scientific | preserve |
| D2_framing_lenient | two‑stage | 0.0 | gpt‑4.1 | **lenient** | scientific | preserve |
| D2_genre_prose | two‑stage | 0.0 | gpt‑4.1 | structured | **prose** | preserve |
| D2_unconstrained | two‑stage | 0.0 | gpt‑4.1 | structured | scientific | **none** |

### Models, data, metrics

- **Models.** Reviser/workhorse = **gpt‑4.1**; independent cross‑model reviewer and
  LLM‑judge = **gpt‑4o** (a different model generation, honouring the "judge ≠
  reviser" self‑bias caution of Xu et al. 2024). Embeddings =
  `text-embedding-3-small`. Decoding: `max_tokens = 2048`, `seed = 42`, temperature
  as tabled. *(Deviation from the plan's Claude workhorse: only `OPENAI_API_KEY`
  was available in this environment; the loud‑fail policy in `llm.py` prevents
  silent mocking.)*
- **Data.** 12 fresh arXiv abstracts spread across 6 categories
  (`datasets/documents/scientific_abstracts.json`) for scientific conditions; 10
  Wikipedia intros (`general_prose.json`) for the genre ablation. Fresh 2025‑era
  documents reduce training‑data‑contamination risk.
- **Metrics** (mirroring and extending Wu et al. 2026):
  - Per‑step normalized Levenshtein `Δt = lev(V_t,V_{t+1}) / max(|V_t|,|V_{t+1}|)`.
  - **Exact** FP (`V_{t+1}=V_t`); **approximate** FP (`Δt < 0.02` and relative
    word‑count change `< 0.03` for 2 consecutive steps).
  - Exponential‑relaxation fit `Δt ≈ A e^{-kt} + c` (report floor `c`, `R²`).
  - Semantic drift `cos(emb(V_0), emb(V_t))`; document length trajectory.
  - LLM‑judge (gpt‑4o) absolute quality (clarity, conciseness, technical precision,
    1–10) at V₀/V_mid/V_final, plus meaning‑preservation and added/dropped‑claim
    counts of V_final vs V₀.
- **Truncation control.** Because runaway length inflation (below) pushes revised
  documents past the 2048‑token cap, a step is flagged "truncated" if either
  endpoint version was cut off (`finish_reason == 'length'`). We report a robustness
  metric — the minimum Δt over **truncation‑free** steps only — to show the
  non‑convergence is not a truncation artifact.
- **Cost / reproducibility.** All chat and embedding calls are disk‑cached
  (`results/cache/`, 2076 entries), so the entire study re‑runs at $0 and is
  crash‑resumable. Full re‑analysis reported `cache_hits = 1222, est_usd = 0.0`.
  Raw trajectories: `results/trajectories/*.json`.

---

## 4. Results

### 4.1 Primary result — the two‑stage loop does not converge

| Condition | mean final Δ | clean min Δ† | % exact FP | % approx FP | fit floor c | fit R² | mean truncated versions |
|---|---|---|---|---|---|---|---|
| **D1_main** (two‑stage) | 0.376 | 0.340 | 0.00 | 0.00 | 0.307 | 0.61 | 3.9 |
| **D2_single_stage** | **0.048** | **0.046** | **0.08** | **0.42** | **0.048** | **0.98** | 0.0 |
| D2_temp_default | 0.601 | 0.567 | 0.00 | 0.00 | 0.455 | 0.27 | 4.2 |
| D2_cross_model | 0.228 | 0.234 | 0.00 | 0.00 | 0.250 | 0.79 | 2.8 |
| D2_framing_harsh | 0.451 | 0.453 | 0.00 | 0.00 | 0.324 | 0.57 | 5.2 |
| D2_framing_lenient | 0.245 | 0.218 | 0.00 | 0.00 | 0.244 | 0.77 | 2.9 |
| D2_genre_prose | 0.275 | 0.289 | 0.00 | 0.00 | 0.170 | 0.83 | 2.4 |
| D2_unconstrained | 0.399 | 0.473 | 0.00 | 0.00 | 0.384 | 0.52 | 5.4 |

† *clean min Δ* = smallest per‑step Δ over truncation‑free steps (robustness check).

The approximate‑fixed‑point threshold is **0.02**. Every two‑stage condition sits
**an order of magnitude above it** and never reaches it (0% exact, 0% approximate).
The exponential fit for D1 leaves a large positive floor `c ≈ 0.31`: the process
relaxes toward a *moving, non‑zero* level of churn, not toward zero. The
single‑stage baseline, by contrast, decays cleanly to `c ≈ 0.05` (R² = 0.98) and
reaches an approximate fixed point in 42% of documents — replicating Wu et al.
(2026) and confirming our harness can detect convergence when it exists.

The per‑step Δ curves make the contrast concrete (`figures/delta_curves.png`):

```
D1_main (two-stage):  0.639 0.489 0.393 0.372 0.404 0.373 0.358 0.376   -> plateau ~0.38
single_stage:         0.225 0.092 0.063 0.059 0.050 0.049 0.049 0.048   -> decays to ~0.05
temp_default:         0.697 0.623 0.583 0.619 0.671 0.622 0.581 0.601   -> oscillates, no relaxation
```

**Statistical tests** (`results/stats.json`):
- D1 final Δ vs 0: *t*(11) = 11.30, *p* = 2.2×10⁻⁷.
- D1 clean min Δ vs the 0.02 approx‑FP threshold: *t*(11) = 13.01, *p* = 5.1×10⁻⁸
  (non‑convergence survives the truncation control).
- Single‑ vs two‑stage, paired on the same docs (clean min Δ): Wilcoxon *p* =
  4.9×10⁻⁴, **Cohen's dz = −3.26** (final Δ: dz = −2.56). The single‑vs‑two‑stage
  distinction is the dominant effect in the whole study.

### 4.2 Determinants of (non‑)convergence — D2 ablations

Paired comparisons vs D1_main (Wilcoxon on final Δ; `results/stats.json`):

| Ablation | direction | effect (Cohen's dz) | p | reading |
|---|---|---|---|---|
| single‑stage vs two‑stage | ↓↓↓ converges | −2.56 | 5×10⁻⁴ | **the** decisive factor |
| temperature 1.0 vs 0 | ↑ more churn / chaotic | +2.09 | 5×10⁻⁴ | default temp destroys even relaxation (R²=0.27) |
| cross‑model reviewer vs self | ↓ less churn | −1.17 | 5×10⁻³ | a different reviewer eases pressure, still no FP |
| lenient vs structured framing | ↓ less churn | −1.14 | 7×10⁻³ | less to fix → smaller edits, still no FP |
| harsh vs structured framing | ↑ (n.s.) | +0.45 | 0.13 | trend toward more churn |
| unconstrained vs length‑preserving | ↑ (n.s.) | +0.19 | 0.62 | length guidance barely helps (see 4.3) |

No manipulation produces convergence. The dynamics are modulated (a *cross‑model*
or *lenient* reviewer roughly halves the residual churn; a *higher temperature*
makes it worse and non‑monotone), but the fixed point is never reached.

### 4.3 Mechanism — runaway length inflation drives drift, not fidelity

Mean document length (characters), V₀ → V₈ (`figures/length_growth.png`):

| Condition | V₀ | V₈ | growth |
|---|---|---|---|
| D2_single_stage | 1469 | 1519 | **1.03×** |
| D1_main | 1469 | 9869 | **6.72×** |
| D2_temp_default | 1469 | 9340 | 6.36× |
| D2_unconstrained | 1469 | 10478 | 7.13× |
| D2_genre_prose | 703 | 10380 | **14.8×** |

The single‑stage loop holds length; every two‑stage loop **explodes** it. Crucially,
the length‑preserving instruction ("keep roughly the same length") is **largely
ignored** under review pressure — D1 (with the instruction) inflates 6.7× and is
statistically indistinguishable from D2_unconstrained (without it, 7.1×; p = 0.62).
The review stage's steady demand for "more precision / examples / discussion"
overrides the length constraint.

This inflation is what breaks fidelity (`figures/semantic_drift.png`, and
LLM‑judge meaning scores):

| Condition | cos(V₀,V_final) | meaning preservation /10 | added claims | dropped claims |
|---|---|---|---|---|
| D2_single_stage | **0.973** | **10.0** | 0.0 | 0.0 |
| D1_main | 0.769 | 6.2 | 4.2 | 2.1 |
| D2_framing_harsh | 0.735 | 4.8 | 8.4 | 3.6 |
| D2_genre_prose | 0.627 | 4.8 | 15.5 | 1.6 |

**Yet isolated‑quality judge scores barely move** (`figures/quality_trajectory.png`):
D1 clarity/conciseness/precision averages **7.50 → 8.00** from V₀ to V_final —
essentially flat and slightly *up*, even as meaning preservation collapses to 6.2
and ~4 new claims are fabricated per document. The loop looks locally like
improvement while globally diverging from the source: the self‑bias / feedback‑
friction signature.

### 4.4 Qualitative example (D1, `abs_000`)

- **V₀ (1617 chars):** a dense, single‑paragraph arXiv abstract on a
  "capability‑driven data infrastructure" for image generation.
- **V₈ (10479 chars):** a multi‑section mini‑paper with invented scaffolding —
  headings like *"Key Concepts and Definitions"*, bulleted term glossaries,
  operational definitions, examples — none of which were in the original.
- **The review that drove it (R₇):** *"some definitions … could be more precise …
  provide concrete examples for less intuitive terms … streamline sections to avoid
  redundancy."* Each round the reviewer finds fresh, legitimate‑sounding gaps; each
  round the reviser fills them by **adding** material. There is no state in which the
  reviewer returns "no changes needed," so the loop cannot reach a fixed point.

Output files: figures in `figures/`, per‑trajectory metrics in
`results/per_trajectory.{json,csv}`, aggregates in `results/summary.json`,
tests in `results/stats.json`, full text trajectories in `results/trajectories/`.

---

## 5. Analysis & Discussion

**Answer to the hypothesis.** For the two‑stage review→revise loop the answer is a
clear **no** — it does not converge to a fixed point within a practical horizon; it
enters a non‑converging **expansion regime**. The hypothesis *is* essentially true
for the *single‑stage* variant (self‑refine relaxes to a soft fixed point), which
is the crucial contrast: **the explicit, regenerated review is what prevents
convergence.**

**Interpretation.** A "make this better" editor can declare a document
locally optimal and stop editing (Δ→0). A "peer reviewer" role is adversarial by
construction: it is prompted to *find weaknesses*, and a capable model can always
find some. Each critique is new edit pressure, and because the cheapest way to
"address" most critiques is to add explanatory material, the operator `F` behaves
less like a contraction toward a fixed point and more like a **growth map**. This
reconciles the two camps in the literature: Wu et al.'s convergent relaxation holds
for single‑stage self‑refine, while the skeptics' warnings (self‑bias amplification,
feedback friction, recursion‑induced degradation) manifest precisely once an
explicit critique loop is added.

**Effect sizes are large and consistent.** The single‑vs‑two‑stage contrast
(dz ≈ −2.6 to −3.3) dwarfs every within‑two‑stage manipulation. Temperature,
reviewer identity, and framing tune the *rate* of churn but not the *existence* of
a fixed point.

**Surprise.** Absolute quality scores rising slightly while meaning preservation
falls sharply is a concrete demonstration that an LLM‑judge scoring documents *in
isolation* is blind to fidelity drift — a caution for anyone using isolated
LLM‑judge scores as a convergence or quality signal.

---

## 6. Limitations

- **Model / provider scope.** One reviser family (gpt‑4.1) with gpt‑4o as
  cross‑model reviewer and judge, due to key availability. The mechanism (a
  never‑satisfied reviewer role) is model‑general in principle, but the specific
  non‑convergence should be replicated on Claude/Gemini and open models before
  claiming universality.
- **Output‑length cap (2048 tokens).** Runaway inflation causes truncation on later
  two‑stage steps (mean ~3–5 of 8 versions truncated in the worst conditions).
  Truncation could distort raw Δ. We mitigate with the truncation‑free "clean min
  Δ" metric, which still rejects convergence (D1 clean min Δ = 0.34 ≫ 0.02,
  *p* = 5×10⁻⁸); truncation is itself a *downstream symptom* of the inflation, not
  the cause of non‑convergence. Still, a larger cap would give cleaner late‑step
  curves.
- **Horizon T = 8.** We cannot rule out convergence at much larger T; but the D1 Δ
  curve is flat/rising from step ~3, and length is still growing at V₈, so there is
  no sign of approaching a fixed point.
- **Sample size.** 12 documents/condition (10 for prose). Adequate for the very
  large effects observed (all primary tests *p* < 10⁻³) but modest for subtle
  ablation differences (harsh vs structured is n.s.).
- **Judge is itself an LLM.** Meaning‑preservation and quality are LLM‑judged
  (gpt‑4o), not human‑annotated; we reduce self‑bias by using a different generation
  than the reviser, but human validation is future work.
- **Prompt dependence.** Results are for one reasonable Review/Revise prompt pair.
  A review prompt explicitly permitted to say "no changes needed," or a reviser with
  a hard length budget, might converge — an interesting positive‑control follow‑up.

---

## 7. Conclusions & Next Steps

**Conclusion.** Iterating AI *review → revise* does **not** reach a fixed point: the
regenerated critique injects endless edit pressure, the document inflates ~7× and
drifts from its original meaning, and the process settles into a high, non‑zero
churn plateau rather than a stable point. Removing the separate review stage
(single‑stage self‑refine) restores convergence to a soft fixed point, identifying
the explicit review loop — not iteration per se — as the cause of
non‑convergence.

**Recommended follow‑ups.**
1. **Positive control for convergence:** give the reviewer an explicit "return
   *NO CHANGES NEEDED* if the document is adequate" option and/or the reviser a hard
   length budget; measure whether a fixed point re‑appears.
2. **Cross‑provider replication** (Claude, Gemini, open‑weight) to establish
   generality of the never‑satisfied‑reviewer mechanism.
3. **Length‑controlled operator:** enforce token budgets to disentangle "churn from
   genuine rewriting" vs "churn from growth."
4. **Basins / path‑dependence (D4):** re‑seed the same content as paraphrases and
   test whether trajectories share an attractor.
5. **Human evaluation** of meaning drift to validate the LLM‑judge fidelity signal.

---

## References (papers, datasets, tools)

- Wu, Xu, Kang, Chen, Liu, Yin (2026). *Do Language Models Converge to Themselves?
  Recursive Self‑Refinement as Textual Relaxation.* arXiv:2607.22653.
  (`papers/converge_to_themselves.pdf`) — direct precedent; single‑stage relaxation.
- Madaan et al. (2023). *Self‑Refine: Iterative Refinement with Self‑Feedback.*
  (`papers/self_refine.pdf`)
- Huang et al. (2023). *Large Language Models Cannot Self‑Correct Reasoning Yet.*
  (`papers/cannot_self_correct.pdf`)
- Xu et al. (2024). *Pride and Prejudice: LLM Amplifies Self‑Bias in
  Self‑Refinement.* (`papers/self_bias.pdf`) — motivates judge ≠ reviser.
- *Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback* (2025).
  (`papers/feedback_friction.pdf`)
- Shumailov et al. (2023). *The Curse of Recursion.* (`papers/curse_of_recursion.pdf`)
- Datasets: fresh arXiv abstracts + Wikipedia intros
  (`datasets/documents/*.json`; see `datasets/README.md`).
- Tools: OpenAI Python SDK (gpt‑4.1, gpt‑4o, text‑embedding‑3‑small),
  `python-Levenshtein`, `scipy`, `numpy`, `matplotlib`. Harness in `code/`.

*Full literature synthesis: `literature_review.md`. Resource catalog: `resources.md`.*
