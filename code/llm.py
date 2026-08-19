"""
LLM wrapper for the Fixed-Points-of-AI-Revision experiments.

- Uses the OpenAI API (OPENAI_API_KEY). OpenRouter/Anthropic keys are not
  present in this environment, so we run entirely on OpenAI models. To honour
  the "judge != reviser" caution from the literature (Xu et al. 2024 self-bias)
  we use genuinely different model *generations*: gpt-4.1 as the workhorse
  reviser and gpt-4o as the independent cross-model reviewer / judge.
- All chat + embedding calls are cached to disk (keyed by a hash of the full
  request) so the whole study is reproducible and re-runnable at ~zero extra
  cost, and so a crash mid-run resumes instantly.

Fails loudly if the API key is missing (no silent mocking).
"""
import os
import json
import time
import hashlib
import threading

import openai

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. This experiment requires a real LLM API; "
        "refusing to run with mocked outputs."
    )

_client = openai.OpenAI(api_key=_API_KEY)
_lock = threading.Lock()

# running token counters (for cost reporting)
TOKENS = {"prompt": 0, "completion": 0, "embed": 0, "calls": 0,
          "cache_hits": 0, "truncated": 0}


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{kind}_{key}.json")


def chat(system: str, user: str, model: str = "gpt-4.1",
         temperature: float = 0.0, max_tokens: int = 2048,
         seed: int = 42) -> str:
    """Single-turn chat completion with disk cache. Returns the text content."""
    payload = {
        "model": model, "system": system, "user": user,
        "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
    }
    key = _key(payload)
    path = _cache_path("chat", key)
    if os.path.exists(path):
        with _lock:
            TOKENS["cache_hits"] += 1
        with open(path) as f:
            return json.load(f)["content"]

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    last_err = None
    for attempt in range(6):
        try:
            resp = _client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, seed=seed,
            )
            content = resp.choices[0].message.content or ""
            truncated = resp.choices[0].finish_reason == "length"
            with _lock:
                TOKENS["prompt"] += resp.usage.prompt_tokens
                TOKENS["completion"] += resp.usage.completion_tokens
                TOKENS["calls"] += 1
                if truncated:
                    TOKENS["truncated"] += 1
            with open(path, "w") as f:
                json.dump({"content": content, "payload": payload,
                           "finish_reason": resp.choices[0].finish_reason}, f)
            return content
        except Exception as e:  # noqa: BLE001 - retry on any transient API error
            last_err = e
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
    raise RuntimeError(f"chat() failed after retries: {last_err}")


def embed(text: str, model: str = "text-embedding-3-small") -> list:
    """Embedding with disk cache."""
    payload = {"model": model, "text": text}
    key = _key(payload)
    path = _cache_path("embed", key)
    if os.path.exists(path):
        with _lock:
            TOKENS["cache_hits"] += 1
        with open(path) as f:
            return json.load(f)["embedding"]
    for attempt in range(6):
        try:
            resp = _client.embeddings.create(model=model, input=[text])
            emb = resp.data[0].embedding
            with _lock:
                TOKENS["embed"] += resp.usage.total_tokens
                TOKENS["calls"] += 1
            with open(path, "w") as f:
                json.dump({"embedding": emb, "payload": payload}, f)
            return emb
        except Exception as e:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("embed() failed after retries")


def cost_report() -> dict:
    """Approximate USD cost from token counters (list prices, Aug 2025)."""
    # gpt-4.1: $2/M in, $8/M out ; gpt-4o: $2.5/M in, $10/M out (blended approx)
    in_cost = TOKENS["prompt"] / 1e6 * 2.5
    out_cost = TOKENS["completion"] / 1e6 * 9.0
    emb_cost = TOKENS["embed"] / 1e6 * 0.02
    total = in_cost + out_cost + emb_cost
    return {**TOKENS, "est_usd": round(total, 3)}
