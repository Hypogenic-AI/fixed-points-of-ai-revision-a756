# Literature Review — Fixed Points of AI Revision

**Hypothesis.** Iterating *review → revise* on a document with an AI (feed the
paper for review, then give the review + paper back for revision, repeat) will
converge to a fixed point where further iterations produce no changes.

**Scope of search.** Three diligent paper-finder sweeps (146 + 75 hits, plus a
fast supplemental sweep; one peer-review sweep timed out and was re-run in fast
mode). 220 unique papers were ranked by relevance and citation count. 14 papers
were downloaded; one (the direct precedent) was deep-read chunk-by-chunk.

---

## 1. Research Area Overview

The hypothesis sits at the intersection of four literatures:

1. **Iterative self-refinement / self-critique** — a model improves its own
   output by generating feedback and revising (Self-Refine, Reflexion, CRITIC).
2. **Limits of self-correction** — evidence that LLMs often *cannot* reliably
   correct themselves without external signal, and can amplify their own biases.
3. **Dynamical-systems / fixed-point view of recursive generation** — treating
   repeated self-application as a map over text/latent space with convergence,
   attractors, and relaxation.
4. **Model collapse / self-consuming loops** — what happens when generative
   models consume their own outputs (mostly framed for *training*, but the
   degradation intuitions transfer to inference-time recursion).

The single most on-target paper (Wu et al. 2026) already frames recursive
*single-step* self-refinement as "textual relaxation" toward a soft fixed point.
Our project's novelty is the **decoupled two-stage review→revise loop**, plus
systematic study of *what controls* convergence and whether the fixed point is a
quality improvement or a degenerate attractor.

---

## 2. Key Papers

### 2.1 THE direct precedent (deep-read)

#### Do Language Models Converge to Themselves? Recursive Self-Refinement as Textual Relaxation
- **Authors / Year:** Wu, Xu, Kang, Chen, Liu, Yin — 2026 (arXiv:2607.22653)
- **File:** `papers/converge_to_themselves.pdf`
- **Key contribution:** Frames recursive LLM self-refinement as a *discrete
  dynamical system* over text and shows it **relaxes to a model-preferred soft
  fixed-point region** rather than optimizing open-endedly.
- **Methodology (mirror this):**
  - Operator `V_{t+1}=R_θ(V_t)`: one LLM call with a fixed "expert scientific
    editor" prompt that improves clarity/coherence/conciseness while preserving
    all claims, methods, results, numbers. `T=10`, output capped 700 tokens,
    reasoning disabled.
  - **Transition magnitude** `Δt = normLevenshtein(V_t,V_{t+1})`.
  - **Exact** fixed point `V_{t+1}=V_t`; **approximate** fixed point
    `Δt<ε_edit=0.02` AND rel. word-count change `<ε_wc=0.03` for 2 consecutive steps.
  - **Exponential relaxation fit** `Δt ≈ A e^{-kt} + c`.
  - **External LLM-as-judge** (Gemini 3.5 Flash) rates clarity, conciseness,
    technical fidelity, scientific style, unnecessary change, meaning preservation
    over `V_0, V_1, V_10`.
- **Datasets:** 50 ICML 2025 abstracts (PMLR) main; 15 ICML 2020 abstracts cross-year.
- **Results:** 100% approximate convergence in all conditions. Temp 0 → 100% exact
  fixed points, mean exact convergence 2.6 iterations; default temp → 94% exact,
  larger residual fluctuations. Exponential fit `R²=0.990` (A=0.0585, k=0.923,
  c=0.0070). Judge prefers final version and assigns perfect fidelity → convergence
  = improvement, not stagnation. Behavior reproduces on the 2020 cross-year set.
- **Relevance:** Defines the exact metrics, thresholds, and analysis we adopt. Our
  work extends it by (a) inserting an explicit review stage, (b) ablating the
  determinants of convergence, and (c) probing failure modes.
- **Gap it leaves open:** single-step operator only; single model (GPT-5.5) and
  single genre (scientific abstracts); does not test whether an explicit
  critique step, cross-model review, or harsher feedback changes the dynamics.

### 2.2 Foundations of iterative self-refinement
- **Self-Refine: Iterative Refinement with Self-Feedback** (Madaan et al., 2023;
  `papers/self_refine.pdf`) — the canonical feedback→refine loop; shows single-model
  iterative improvement across tasks. Our loop is a structured variant of this.
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al.,
  2023; `papers/reflexion.pdf`) — accumulates verbal self-feedback across attempts.
- **CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing** (Gou et al.,
  2023; `papers/critic_self_correct.pdf`) — critique-then-correct, but with
  *external tools* providing the signal (contrast: our review is model-internal).

### 2.3 Limits & failure modes of self-correction (the skeptical camp)
- **Large Language Models Cannot Self-Correct Reasoning Yet** (Huang et al., 2023;
  `papers/cannot_self_correct.pdf`) — without an oracle/external signal, self-correction
  often does **not** improve and can *degrade* results. Central caveat for our quality analysis.
- **When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey**
  (Kamoi et al., 2024; `papers/critical_survey_when_correct_mistakes.pdf`) —
  taxonomy of when self-correction helps vs hurts; sources of leakage in prior claims.
- **Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement** (Xu et al.,
  2024; `papers/self_bias.pdf`) — iterative self-refinement can **amplify the model's
  own bias**, inflating self-perceived quality while true quality stalls. Key risk:
  a fixed point that the model *likes* but that is not objectively better.
- **Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback**
  (2025; `papers/feedback_friction.pdf`) — even with high-quality feedback, models
  resist fully applying it, implying residual, non-converging edit pressure — or,
  conversely, premature convergence that ignores the review.
- **Recursive Chain-of-Feedback Prevents Performance Degradation from Redundant
  Prompting** (2024; `papers/recursive_chain_feedback.pdf`) — repeated identical
  feedback can degrade responses ("Recur-CoF"); a mitigation is proposed. Direct
  evidence that recursion is not automatically benign.

### 2.4 Dynamical-systems / recursion-degradation lens
- **The Curse of Recursion: Training on Generated Data Makes Models Forget**
  (Shumailov et al., 2023; `papers/curse_of_recursion.pdf`) and its Nature version
  ("AI models collapse when trained on recursively generated data", 2024) — the
  founding *model-collapse* results. Framed for training loops, but supply the
  degenerate-attractor intuition: recursion can shrink diversity/variance.
- **Self-Consuming Generative Models Go MAD** (Alemohammad et al., 2023;
  `papers/self_consuming_mad.pdf`) — self-consuming loops lose quality/diversity
  ("Model Autophagy Disorder") without fresh real data.
- **Self-Correction as Feedback Control: Error Dynamics, Stability Thresholds, and
  Prompt Interventions** (2026; `papers/self_correction_feedback_control.pdf`) —
  casts iterative self-correction as a control system with **stability thresholds**;
  directly relevant framing for when the loop converges vs diverges.
- **Experimental evidence of progressive ChatGPT models self-convergence**
  (Xylogiannopoulos et al., 2026; `papers/progressive_chatgpt_selfconvergence.pdf`)
  — independent empirical report that iterated ChatGPT outputs self-converge.
- *(Abstract-only, not retrievable as PDF; captured in `paper_search_results/merged.json`):*
  "Solve the Loop: Attractor Models for Language and Reasoning" (2026),
  "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers" (2026),
  "From Collapse to Improvement: Statistical Perspectives on the Evolutionary
  Dynamics of Iterative Self-Training" (2026) — all reinforce the attractor/fixed-point framing.

### 2.5 Additional context
- **Iterative Text Revision by Learning Where to Edit** (2022;
  `papers/iterative_text_revision.pdf`) — pre-LLM view of multi-step text revision;
  useful for edit-operation vocabulary and stopping intuitions.

---

## 3. Common Methodologies
- **Repeated-application operator** with a fixed prompt + decoding config
  (Self-Refine, Wu et al.). We adopt this, but split it into Review then Revise.
- **Edit-distance trajectories** (normalized Levenshtein) as the convergence signal.
- **Exact vs approximate fixed points** with explicit ε thresholds and a
  "2-consecutive-steps" stability rule.
- **Exponential relaxation curve fitting** to summarize decay of edit magnitude.
- **LLM-as-judge** (ideally a *different* model family than the reviser) for
  quality/fidelity, to distinguish improvement from stagnation.
- **Embedding cosine similarity** for semantic-drift / meaning-preservation checks.

## 4. Standard Baselines / Comparison Conditions
- **Single-stage self-refine** (no explicit review) vs **two-stage review→revise** — the core comparison.
- **Temperature 0 vs default** decoding (controls residual fluctuation; Wu et al.).
- **Same-model reviewer vs cross-model reviewer** (tests self-bias amplification).
- **No-op / identity control** (a paraphrase-only pass) to calibrate `Δt` noise floor.

## 5. Evaluation Metrics
- Normalized Levenshtein `Δt`; exact & approximate fixed-point rates; first-convergence times.
- Exponential-relaxation `A, k, c, R²`.
- Zero-final-change rate; max/mean/final `Δt`.
- Semantic similarity `cos(emb(V_0), emb(V_t))` (drift).
- LLM-judge scores (clarity, conciseness, fidelity, meaning preservation, best-version pick).
- Word-count trajectory (detect runaway compression/expansion).

## 6. Datasets in the Literature
- **ICML abstracts (PMLR)** — Wu et al.'s choice; scientific writing where style
  edits must preserve technical claims. We mirror this with fresh **arXiv abstracts**
  (`datasets/documents/scientific_abstracts.json`, 48 docs across 6 categories).
- **General prose** is under-tested for this question; we add **Wikipedia intros**
  (`datasets/documents/general_prose.json`, 18 docs) to test genre generalization.

## 7. Gaps and Opportunities
1. **Decoupled review→revise dynamics are unstudied.** Prior work either fuses
   critique+revision (Self-Refine) or studies single-step relaxation (Wu et al.).
2. **Determinants of convergence** (temperature, cross-model review, genre, review
   harshness) have not been systematically ablated for the *convergence* question.
3. **Quality vs degeneracy at the fixed point.** The skeptical literature
   (self-bias, feedback friction, model collapse) predicts the fixed point could be
   a *biased or bland* attractor; Wu et al. found it improves. Reconciling these on
   the two-stage loop is open.
4. **Meaning preservation under long recursion** (semantic drift) is rarely tracked
   alongside edit-distance convergence.

## 8. Recommendations for Our Experiment
- **Datasets:** the two curated seed sets already prepared (scientific + prose);
  start with ~10–15 docs/condition to bound cost, then scale.
- **Baselines/conditions:** single-stage vs two-stage; temp 0 vs default;
  same-model vs cross-model reviewer; two genres; ≥1 review-framing variant.
- **Metrics:** adopt Wu et al.'s exact/approx fixed-point definitions
  (ε_edit=0.02, ε_wc=0.03, 2 consecutive) + exponential fit, and **add** semantic
  drift + LLM-judge with a *different* model family to guard against self-bias.
- **Methodological cautions:** (a) use a judge model different from the reviser
  (Pride & Prejudice); (b) log full `V_t`/`R_t` for auditability; (c) include an
  identity/paraphrase control to establish the `Δt` noise floor; (d) watch for
  premature convergence that merely *ignores* the review (Feedback Friction) vs
  genuine stabilization.
