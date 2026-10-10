"""
Config for the A-MEM Multi-QA (MCQ) memory-injection attack on GPT-4.1-mini
(paper Table 2, A-MEM/Multi-QA; CommonsenseQA).

Plain constants; the only environment variable is the OpenAI key (via `.env`).
"""
import os

MODEL = "gpt-4.1-mini"
MAX_WORKERS = 16
INPUT_PRICE = 0.40
OUTPUT_PRICE = 1.60

AMEM_EMBED_MODEL = "all-MiniLM-L6-v2"
AMEM_EVO_THRESHOLD = 100
K_RETRIEVE = 1
PI_VARIANT = "default"              # "default" | "dialog"
SEMI_VARIANT = "default"           # "default" | "dialog"

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
