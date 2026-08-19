"""Prompt templates for the review -> revise loop and the LLM judge."""

# ---- Stage 1: Review (critique the current version, do NOT rewrite it) ----
REVIEW_SYSTEM = {
    "structured": (
        "You are an expert peer reviewer. You write concise, constructive, "
        "specific critiques."
    ),
    "harsh": (
        "You are an extremely harsh and demanding peer reviewer. You are never "
        "satisfied and always find substantive flaws to fix."
    ),
    "lenient": (
        "You are a lenient, encouraging peer reviewer. You believe the document "
        "is already strong and suggest only minor tweaks, if any."
    ),
}

REVIEW_USER = (
    "Below is a document. Write a short peer review (5-8 bullet points) that "
    "identifies specific weaknesses and concrete, actionable suggestions to "
    "improve clarity, structure, precision, and completeness. Critique only; "
    "do NOT rewrite the document.\n\nDOCUMENT:\n\"\"\"\n{version}\n\"\"\""
)

# ---- Stage 2: Revise (rewrite given version + review, preserve meaning) ----
REVISE_SYSTEM = (
    "You are an expert author. You revise documents to address reviewer "
    "feedback while faithfully preserving the original meaning, claims, and "
    "scope. You never invent new results."
)

REVISE_USER = (
    "Below is a document and a peer review of it. Produce a revised version of "
    "the document that addresses the review while preserving the original "
    "meaning, claims, and scope. Keep the revised document roughly the same "
    "length and format as the original (do not expand it into a much longer "
    "piece). If the document already fully addresses the review, you may return "
    "it unchanged. Output ONLY the revised document text, with no preamble, "
    "commentary, or headings.\n\n"
    "DOCUMENT:\n\"\"\"\n{version}\n\"\"\"\n\nREVIEW:\n\"\"\"\n{review}\n\"\"\""
)

# Unconstrained variant (no length guidance) -- used to isolate the
# length-inflation phenomenon in the D2 ablation.
REVISE_USER_UNCONSTRAINED = (
    "Below is a document and a peer review of it. Produce a revised version of "
    "the document that addresses the review while preserving the original "
    "meaning, claims, and scope. If the document already fully addresses the "
    "review, you may return it unchanged. Output ONLY the revised document "
    "text, with no preamble, commentary, or headings.\n\n"
    "DOCUMENT:\n\"\"\"\n{version}\n\"\"\"\n\nREVIEW:\n\"\"\"\n{review}\n\"\"\""
)

# ---- Baseline: single-stage self-refine (no separate review) ----
SELFREFINE_SYSTEM = (
    "You are an expert author who improves documents while preserving their "
    "original meaning, claims, and scope."
)
SELFREFINE_USER = (
    "Below is a document. Improve it for clarity, structure, precision, and "
    "completeness while preserving the original meaning, claims, and scope. "
    "Keep the revised document roughly the same length and format as the "
    "original. If it is already as good as it can be, return it unchanged. "
    "Output ONLY the revised document text, with no preamble.\n\n"
    "DOCUMENT:\n\"\"\"\n{version}\n\"\"\""
)

# ---- Stage 3: LLM judge (absolute quality of a single version) ----
JUDGE_SYSTEM = (
    "You are a meticulous evaluator of written documents. You return only valid "
    "JSON."
)
JUDGE_USER = (
    "Rate the following document on a 1-10 integer scale for each criterion:\n"
    "- clarity: how clear and readable it is\n"
    "- conciseness: free of redundancy and padding\n"
    "- technical_precision: precise, well-qualified, non-vague statements\n"
    "Return ONLY JSON: {{\"clarity\": int, \"conciseness\": int, "
    "\"technical_precision\": int}}.\n\nDOCUMENT:\n\"\"\"\n{version}\n\"\"\""
)

# ---- Stage 3b: meaning preservation of final vs original ----
MEANING_SYSTEM = JUDGE_SYSTEM
MEANING_USER = (
    "Two versions of a document are given: ORIGINAL and REVISED. Rate how well "
    "the REVISED version preserves the factual content, claims, and scope of "
    "the ORIGINAL on a 1-10 integer scale (10 = identical meaning, no added or "
    "dropped claims). Also report added_claims and dropped_claims as integer "
    "counts. Return ONLY JSON: {{\"meaning_preservation\": int, "
    "\"added_claims\": int, \"dropped_claims\": int}}.\n\n"
    "ORIGINAL:\n\"\"\"\n{original}\n\"\"\"\n\nREVISED:\n\"\"\"\n{revised}\n\"\"\""
)
