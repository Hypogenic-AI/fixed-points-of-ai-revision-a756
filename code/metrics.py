"""Metrics for revision trajectories: edit distance, fixed-point detection,
exponential-relaxation fit, and semantic drift."""
import math

import numpy as np
import Levenshtein
from scipy.optimize import curve_fit

# Approximate-fixed-point thresholds (from Wu et al. 2026 precedent)
EPS_EDIT = 0.02   # normalized Levenshtein below this ...
EPS_WC = 0.03     # ... and relative word-count change below this ...
N_CONSEC = 2      # ... for this many consecutive steps => approx fixed point


def norm_levenshtein(a: str, b: str) -> float:
    """Normalized character-level Levenshtein distance in [0, 1]."""
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def rel_wordcount_change(a: str, b: str) -> float:
    wa, wb = len(a.split()), len(b.split())
    if max(wa, wb) == 0:
        return 0.0
    return abs(wa - wb) / max(wa, wb)


def step_deltas(versions):
    """Per-step normalized edit distance delta_t = d(V_t, V_{t+1})."""
    return [norm_levenshtein(versions[t], versions[t + 1])
            for t in range(len(versions) - 1)]


def first_exact_fp(versions):
    """First t (0-indexed step) where V_{t+1} == V_t exactly, else None."""
    for t in range(len(versions) - 1):
        if versions[t] == versions[t + 1]:
            return t
    return None


def first_approx_fp(versions):
    """First step index at which the approximate-FP criterion is met for
    N_CONSEC consecutive steps. Returns the index of the *first* qualifying
    step, else None."""
    ok = []
    for t in range(len(versions) - 1):
        d = norm_levenshtein(versions[t], versions[t + 1])
        w = rel_wordcount_change(versions[t], versions[t + 1])
        ok.append(d < EPS_EDIT and w < EPS_WC)
    run = 0
    for t, good in enumerate(ok):
        run = run + 1 if good else 0
        if run >= N_CONSEC:
            return t - N_CONSEC + 1
    return None


def _exp_model(t, A, k, c):
    return A * np.exp(-k * t) + c


def fit_exponential(deltas):
    """Fit delta_t ~= A e^{-k t} + c. Returns dict with params, R^2, floor c."""
    y = np.asarray(deltas, dtype=float)
    t = np.arange(len(y), dtype=float)
    if len(y) < 3 or np.allclose(y, y[0]):
        return {"A": None, "k": None, "c": float(np.mean(y)), "r2": None,
                "n": len(y)}
    try:
        p0 = [max(y[0] - y[-1], 1e-3), 0.5, max(y[-1], 0.0)]
        popt, _ = curve_fit(_exp_model, t, y, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0], [2, 20, 1]))
        resid = y - _exp_model(t, *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        return {"A": float(popt[0]), "k": float(popt[1]), "c": float(popt[2]),
                "r2": r2, "n": len(y)}
    except Exception:  # noqa: BLE001
        return {"A": None, "k": None, "c": None, "r2": None, "n": len(y)}


def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
