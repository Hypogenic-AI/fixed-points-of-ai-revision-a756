# Fixed Points of AI Revision

Does iterating **AI review → AI revise** on a document converge to a fixed point
where further iterations change nothing? We run the loop
`R_t = Review(V_t); V_{t+1} = Revise(V_t, R_t)` for 8 iterations on 12 scientific
abstracts (+ ablations) with real OpenAI models and measure convergence.

## Key findings

- **No fixed point for the two‑stage review→revise loop.** Per‑step normalized edit
  distance plateaus at **Δ ≈ 0.38** (approx‑fixed‑point threshold = 0.02); **0%** of
  trajectories converge (exact or approximate). One‑sample *t* = 11.3, *p* = 2×10⁻⁷.
- **Single‑stage self‑refine *does* converge** on the same documents (mean final
  Δ ≈ 0.05, 42% reach an approximate fixed point, exp‑relaxation R² = 0.98),
  replicating Wu et al. (2026). The **explicit review stage is what prevents
  convergence** (paired Cohen's dz = −3.26, *p* = 5×10⁻⁴) — the single dominant
  effect in the study.
- **Mechanism: runaway length inflation.** The never‑satisfied reviewer keeps
  demanding additions, so documents grow **~7×** (a 1.5 KB abstract → ~10 KB
  mini‑paper; prose up to 15×), even with a "keep the same length" instruction.
- **Fidelity degrades while quality *looks* fine.** Meaning preservation falls 10 → 6.2/10
  (~4 fabricated claims/doc) and cosine(V₀,V_final) drops 0.97 → 0.77, yet the
  isolated‑quality judge score barely moves (7.5 → 8.0) — a self‑bias signature.
- **Determinants:** default temperature makes it worse and non‑monotone (R² = 0.27);
  a cross‑model or lenient reviewer roughly halves the churn but still never converges.

See **[REPORT.md](REPORT.md)** for full methodology, tables, statistics, and figures.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync   # deps in pyproject.toml
export OPENAI_API_KEY=...                          # required (fails loudly if unset)

# 1. Run the review->revise trajectories (all LLM calls are disk-cached)
python code/experiments.py                         # -> results/trajectories/*.json

# 2. Analyze: metrics, stats, figures (judge=gpt-4o)
python code/analyze.py                             # -> results/*.json/.csv, figures/*.png
```

All chat/embedding calls are cached in `results/cache/`, so a re‑run costs **$0**
and is crash‑resumable (analysis re‑run reports `est_usd = 0.0`).

## Repository layout

```
code/
  prompts.py       Review / Revise / self-refine / judge prompt templates
  llm.py           Cached OpenAI wrapper (gpt-4.1 reviser, gpt-4o judge/reviewer)
  run_loop.py      One review->revise (or single-stage) trajectory
  experiments.py   8 conditions (D1 + 7 ablations), parallel over docs
  metrics.py       Normalized Levenshtein, exact/approx FP, exp-relaxation fit
  analyze.py       Drift, LLM-judge quality, statistics, figures
datasets/documents/  scientific_abstracts.json, general_prose.json
results/
  trajectories/    Raw V_t and R_t per condition
  per_trajectory.{json,csv}, summary.json, stats.json
  cache/           Disk cache of all API calls (reproducibility)
figures/           delta_curves, length_growth, final_delta_box,
                   clean_min_delta_box, semantic_drift, quality_trajectory (.png)
planning.md, literature_review.md, resources.md, REPORT.md, STATE.md
```

## Conditions

`D1_main` (two‑stage, temp 0, self‑review, scientific) is the direct test.
Ablations: `single_stage`, `temp_default` (temp 1.0), `cross_model` (gpt‑4o
reviewer), `framing_harsh`, `framing_lenient`, `genre_prose`, `unconstrained`
(no length guidance).

## Bottom line

Iterating AI review→revise **does not** reach a fixed point — it enters a
non‑converging expansion regime that inflates and drifts. If you need it to stop,
drop the separate review stage or add an explicit stop criterion.
