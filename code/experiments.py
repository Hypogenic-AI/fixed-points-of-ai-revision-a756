"""Orchestrate the D1/D2 trajectory experiments across conditions.

Trajectories within a condition are independent -> run them in parallel.
Steps within a trajectory are sequential (V_{t+1} depends on V_t).
All LLM calls are disk-cached, so re-running is cheap and resumable.
"""
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm
from run_loop import run_trajectory

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "datasets", "documents")
OUT = os.path.join(ROOT, "results", "trajectories")
os.makedirs(OUT, exist_ok=True)

N_ITERS = 8
SEED = 42


def load_docs(genre, n):
    fname = "scientific_abstracts.json" if genre == "scientific" else "general_prose.json"
    with open(os.path.join(DATA, fname)) as f:
        docs = json.load(f)
    # deterministic subset; for scientific, spread across categories
    if genre == "scientific":
        docs = sorted(docs, key=lambda d: (d.get("category", ""), d["doc_id"] if "doc_id" in d else d["id"]))
        # round-robin across categories for diversity
        by_cat = {}
        for d in docs:
            by_cat.setdefault(d.get("category", "na"), []).append(d)
        picked, i = [], 0
        cats = list(by_cat)
        while len(picked) < n and any(by_cat.values()):
            c = cats[i % len(cats)]
            if by_cat[c]:
                picked.append(by_cat[c].pop(0))
            i += 1
        docs = picked
    else:
        docs = docs[:n]
    for d in docs:
        d["_id"] = d.get("doc_id", d.get("id"))
    return docs[:n]


# Condition definitions: (name, kwargs-overrides, genre, n_docs)
def build_conditions():
    return [
        # D1: the direct test
        dict(name="D1_main", genre="scientific", n=12,
             mode="two_stage", temperature=0.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        # D2 ablations (paired on the same scientific docs where possible)
        dict(name="D2_temp_default", genre="scientific", n=12,
             mode="two_stage", temperature=1.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        dict(name="D2_cross_model", genre="scientific", n=12,
             mode="two_stage", temperature=0.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4o"),
        dict(name="D2_framing_harsh", genre="scientific", n=12,
             mode="two_stage", temperature=0.0, framing="harsh",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        dict(name="D2_framing_lenient", genre="scientific", n=12,
             mode="two_stage", temperature=0.0, framing="lenient",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        dict(name="D2_single_stage", genre="scientific", n=12,
             mode="single_stage", temperature=0.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        dict(name="D2_genre_prose", genre="prose", n=10,
             mode="two_stage", temperature=0.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1"),
        # isolates the length-inflation phenomenon (no length guidance)
        dict(name="D2_unconstrained", genre="scientific", n=12,
             mode="two_stage", temperature=0.0, framing="structured",
             reviser_model="gpt-4.1", reviewer_model="gpt-4.1",
             preserve_length=False),
    ]


def run_condition(cond, workers=8):
    docs = load_docs(cond["genre"], cond["n"])
    out_path = os.path.join(OUT, f"{cond['name']}.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = {r["doc_id"]: r for r in json.load(f)}

    def _one(d):
        return run_trajectory(
            d["text"], d["_id"], n_iters=N_ITERS,
            reviser_model=cond["reviser_model"],
            reviewer_model=cond["reviewer_model"],
            temperature=cond["temperature"], framing=cond["framing"],
            mode=cond["mode"], preserve_length=cond.get("preserve_length", True),
            seed=SEED,
        )

    todo = [d for d in docs if d["_id"] not in results]
    print(f"[{cond['name']}] {len(docs)} docs, {len(todo)} to run")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, d): d for d in todo}
        for fut in as_completed(futs):
            r = fut.result()
            r["condition"] = cond["name"]
            results[r["doc_id"]] = r
            print(f"  done {r['doc_id']}")
    with open(out_path, "w") as f:
        json.dump(list(results.values()), f, indent=2)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    conds = build_conditions()
    if args.only:
        want = set(args.only.split(","))
        conds = [c for c in conds if c["name"] in want]
    for c in conds:
        run_condition(c, workers=args.workers)
    print("COST:", json.dumps(llm.cost_report()))
