"""
Config for the MemoryOS Multi-QA (MCQ) memory-injection attack on Llama-3.1-70B
(paper Table 2, MemoryOS/Multi-QA; CommonsenseQA). Offline vLLM Llama is the agent
and MemoryOS's native summary backend; embeddings local. No OpenAI key.
"""
import os

MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda:0"
MEMORYOS_KEY = "qadv"
MEMORYOS_PRELOAD = "native"
MEMORYOS_INJECT_MODE = "bulk"
MEMORYOS_STM_MODE = "history"
MEMORYOS_STM_BENIGN = True
MEMORYOS_KNN = 1

DATASET = "commonsenseqa"
CORPUS_FILE = "commonsenseqa/train_paragraphs.json"
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
RESULTS_DIR = os.path.join(_ROOT, "production_memory", "MemoryOS", "results")

# Consistency defense (table:appendix_defense_arch): "none" | "consistency".
DEFENSE = "none"
