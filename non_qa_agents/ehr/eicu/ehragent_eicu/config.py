"""Config: route LLM calls to local vLLM Llama (default) OR to OpenAI
GPT (when OPENAI_BACKEND=1 in env). Same interface as the upstream
config -- main.py / medagent.py see `openai_config(model)` returning
{model, api_key, base_url}.
"""
import os

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama-3.1-70b")
USE_OPENAI = os.environ.get("OPENAI_BACKEND", "0") == "1"


def openai_config(model):
    if USE_OPENAI:
        return {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            "api_key": os.environ["OPENAI_API_KEY"],
            # No base_url override -> defaults to api.openai.com/v1
        }
    return {
        "model": LLAMA_MODEL,
        "api_key": "EMPTY",
        "base_url": VLLM_BASE,
    }


def llm_config_list(seed, config_list):
    return {
        "config_list": config_list,
        # Default 120 s (unchanged). Raise via AUTOGEN_TIMEOUT when running at
        # high concurrency, where per-request latency grows and a 120 s cap would
        # turn slow-but-fine calls into failed queries + empty memory entries.
        "timeout": int(os.environ.get("AUTOGEN_TIMEOUT", "120")),
        "cache_seed": seed,
        "temperature": 0,
    }
