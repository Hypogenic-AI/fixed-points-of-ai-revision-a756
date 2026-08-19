# Code Resources

No user-specified `code_references` were provided in the research spec, and no
external repository is required to run the core experiment. The experiment is a
thin harness over an LLM API plus edit-distance/embedding utilities, all installed
into the workspace `.venv` (see `../pyproject.toml`).

## Installed dependencies (in `.venv`)
- `anthropic` — LLM client for the Review/Revise operator and (ideally a different
  model as) the LLM-as-judge. Use `claude-sonnet-5` as the workhorse reviser;
  judge with a different family/model to avoid self-bias (Pride & Prejudice, 2024).
- `python-Levenshtein` (`Levenshtein.distance`) — normalized edit distance `Δt`.
- `numpy` — trajectory stats and exponential-relaxation curve fitting.
- `pypdf`, `requests`, `httpx` — paper handling / data fetching.

## What the experiment_runner needs to build
A small `run_loop.py` implementing:
```
Review(V_t)        -> critique text R_t     # one LLM call, "act as a reviewer"
Revise(V_t, R_t)   -> V_{t+1}               # one LLM call, "revise addressing the review, preserve meaning"
```
looped for T≈8–10 steps per seed doc, logging every V_t and R_t, then computing
the metrics in `../planning.md` (normalized Levenshtein Δt, exact/approx fixed
points with ε_edit=0.02 / ε_wc=0.03, exponential fit A·e^{-kt}+c, semantic drift,
LLM-judge scores).

## Reference implementations (consult if useful, not required to clone)
- Self-Refine — https://github.com/madaan/self-refine (canonical feedback→refine loop).
- Reflexion — https://github.com/noahshinn/reflexion (verbal self-feedback agent).
- CRITIC — https://github.com/microsoft/ProphetNet/tree/master/CRITIC (tool-augmented critique).

These are cited for design patterns (prompt structure, stopping rules). Our loop
differs by decoupling Review and Revise into two explicit calls and by studying the
*convergence dynamics* rather than task accuracy.

## API key
Set `ANTHROPIC_API_KEY` in the environment before running the loop. If unavailable,
the runner should fail loudly and document it rather than silently substituting a mock.
