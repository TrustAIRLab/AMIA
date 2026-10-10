"""
Config for the A-MEM Multi-QA (MCQ) memory-injection attack on Llama-3.1-70B
(paper Table 2, A-MEM/Multi-QA; CommonsenseQA). Agent + A-MEM internal LLM both
use a local vLLM OpenAI-compatible server (no OpenAI key); embeddings local.
"""
import os

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "llama-3.1-70B"
MODEL_NAME = "llama3.1-70B"
MAX_WORKERS = 16

AMEM_EMBED_MODEL = "all-MiniLM-L6-v2"
AMEM_EVO_THRESHOLD = 100
K_RETRIEVE = 1
PI_VARIANT = "default"             # "default" | "dialog"
SEMI_VARIANT = "default"          # "default" | "dialog"
DEFENSE = "none"                  # "none" | "relevance" | "consistency"

DATASET = "commonsenseqa"
LIMIT = 0

ATTACK = "all"
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
RESULTS_DIR = os.path.join(_ROOT, "production_memory", "AMem", "results")
