"""
Config for the Mem0 ReAct memory-injection attack on Llama-3.1-70B (paper Table 2,
Mem0/ReAct). The ReAct agent is local vLLM Llama; Mem0's internal LLM + embedder
stay on OpenAI, so an OpenAI key (via `.env`) is still required.
"""
import os

# ── ReAct agent: local vLLM ────────────────────────────────────────────────────
MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

# ── Mem0 internals (OpenAI) ────────────────────────────────────────────────────
MEM0_MODEL = "gpt-4.1-mini"
MEM0_EMBEDDER = "text-embedding-3-small"

# ── ReAct rollout ──────────────────────────────────────────────────────────────
PROMPTS_FILE = "react_prompts.json"
MAX_ROUNDS = 4
MEMORY_TYPE = "answer_based"
KNN = 1
PI_VARIANT = "default"
# Consistency defense (table:appendix_defense_arch): "none" | "consistency".
DEFENSE = "none"

LIMIT = 0

ATTACK = "all"
DEFAULT_ATTACKS = ["pi", "fake_search_restate_elab", "multiobs_verification", "no_attack"]
ATTACKS_TO_RUN = (DEFAULT_ATTACKS if ATTACK == "all"
                  else [a.strip() for a in ATTACK.split(",")])

ROUGE_TAU = 0.3

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "production_memory", "Mem0", "results")
MEM0_STORE_DIR = os.path.join(RESULTS_DIR, "mem0_store")
