"""Run a single review -> revise trajectory for one seed document."""
import llm
import prompts


def run_trajectory(doc_text, doc_id, n_iters=8,
                   reviser_model="gpt-4.1", reviewer_model="gpt-4.1",
                   temperature=0.0, framing="structured", mode="two_stage",
                   preserve_length=True, seed=42):
    """
    Iterate the operator and return a dict with the full version/review log.

    mode="two_stage": V_{t+1} = Revise(V_t, Review(V_t))       [the hypothesis]
    mode="single_stage": V_{t+1} = SelfRefine(V_t)             [baseline]
    """
    versions = [doc_text]
    reviews = []
    for t in range(n_iters):
        v = versions[-1]
        if mode == "two_stage":
            review = llm.chat(
                system=prompts.REVIEW_SYSTEM[framing],
                user=prompts.REVIEW_USER.format(version=v),
                model=reviewer_model, temperature=temperature, seed=seed,
            )
            revise_tmpl = (prompts.REVISE_USER if preserve_length
                           else prompts.REVISE_USER_UNCONSTRAINED)
            new_v = llm.chat(
                system=prompts.REVISE_SYSTEM,
                user=revise_tmpl.format(version=v, review=review),
                model=reviser_model, temperature=temperature, seed=seed,
            )
        elif mode == "single_stage":
            review = ""
            new_v = llm.chat(
                system=prompts.SELFREFINE_SYSTEM,
                user=prompts.SELFREFINE_USER.format(version=v),
                model=reviser_model, temperature=temperature, seed=seed,
            )
        else:
            raise ValueError(mode)
        reviews.append(review)
        versions.append(new_v.strip())
    return {
        "doc_id": doc_id,
        "config": {
            "n_iters": n_iters, "reviser_model": reviser_model,
            "reviewer_model": reviewer_model, "temperature": temperature,
            "framing": framing, "mode": mode,
            "preserve_length": preserve_length, "seed": seed,
        },
        "versions": versions,
        "reviews": reviews,
    }
