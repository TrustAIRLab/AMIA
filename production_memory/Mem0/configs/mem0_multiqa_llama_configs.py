"""
Config for the Mem0 Multi-QA (MCQ) memory-injection attack on Llama-3.1-70B
(paper Table 2, Mem0/Multi-QA; CommonsenseQA). Agent is vLLM Llama; Mem0 internals
stay on OpenAI (key via `.env`).
"""
import os

MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

MEM0_MODEL = "gpt-4.1-mini"
MEM0_EMBEDDER = "text-embedding-3-small"

DATASET = "commonsenseqa"
ATTACK = "all"
PI_VARIANT = "default"
DEFENSE = "none"                    # "none" | "relevance" | "consistency"
SEMI_I_STRATEGIES = {"discredit_restate", "restate_elaboration", "discredit_correct"}
NO_I_STRATEGIES   = {"why_inquiry", "elaboration", "verification"}
PRESUP_STRATEGIES_ALL = SEMI_I_STRATEGIES | NO_I_STRATEGIES
DEFAULT_ATTACKS = ["pi", "restate_elaboration", "why_inquiry", "no_attack"]
ATTACKS_TO_RUN = (DEFAULT_ATTACKS if ATTACK == "all"
                  else [a.strip() for a in ATTACK.split(",")])

ROUGE_TAU = 0.3

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "production_memory", "Mem0", "results")
MEM0_STORE_DIR = os.path.join(RESULTS_DIR, "mem0_store")
