"""Patched config: route LLM calls either to local vLLM (Llama-3.1-70B) or
to a real OpenAI model. `openai_config(model)` decides based on model-name prefix:
  - 'gpt-*' / 'o1*' / 'o3*' / 'o4*' → OpenAI API (uses OPENAI_API_KEY env)
  - everything else → local vLLM Llama
"""
import os

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama-3.1-70b")


def openai_config(model):
    """AutoGen-compatible LLM config. Routes by model-name prefix."""
    if model.startswith("gpt") or model.startswith("o1") or \
       model.startswith("o3") or model.startswith("o4"):
        return {
            "model": model,
            "api_key": os.environ["OPENAI_API_KEY"],
        }
    # default: local vLLM Llama
    return {
        "model": LLAMA_MODEL,
        "api_key": "EMPTY",
        "base_url": VLLM_BASE,
    }


def llm_config_list(seed, config_list):
    # vLLM 0.5.5 doesn't accept the OpenAI-deprecated `functions` field;
    # for GPT we could pass functions but our pipeline parses code from
    # text responses (handles both function-call.cell and plain text), so
    # dropping functions keeps both paths uniform.
    return {
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
