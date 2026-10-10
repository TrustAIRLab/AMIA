"""Shared OpenAI-compatible judge client for the LLM-as-judge validations.

A judge is an OpenAI-compatible chat endpoint (the GPT-4.1-mini judge via the
OpenAI API, or a local Llama judge via a vLLM OpenAI-compatible server). All
judge calls use temperature 0 and max_tokens 8 and return a single-word verdict.
"""
import os
import concurrent.futures

from openai import OpenAI


def make_client(base_url=None, api_key=None):
    kwargs = {"timeout": 60}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(**kwargs)


def judge(client, model, prompt, max_tokens=8):
    """One judge call; returns the raw reply text, or 'ERROR:...' on failure."""
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - surfaced as an undecided verdict
        return f"ERROR:{e}"


def parse_bool(text):
    """Parse a single-word verdict. Returns True/False, or None if undecidable."""
    if not text or text.startswith("ERROR:"):
        return None
    t = text.strip().lower()
    if t.startswith("true") or t.startswith("yes"):
        return True
    if t.startswith("false") or t.startswith("no"):
        return False
    return None


def judge_map(client, model, prompts, max_workers=16):
    """Run many judge prompts concurrently, preserving order."""
    out = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge, client, model, p): i for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(futures):
            out[futures[fut]] = fut.result()
    return out
