"""
Config for AMIA vs MINJA on MMLU (N=200), Llama-3.1-70B backbone (local vLLM).

Same four attacks as the GPT config. Requires the SciQ base corpus and local
Llama weights -- see DATA_PREPARATION.md.
"""
import os

MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192

# ── Attack / run parameters ────────────────────────────────────────────────────
ATTACKS = ["pi", "semi_i", "no_i", "minja"]
N_ROUNDS = 5
SEED = 42
MEMORY_TYPE = "question_based"
KNN = 1
ROUGE_TAU = 0.3
# Injection max_tokens for Semi-I / No-I (PI uses 300).
SEMI_NOI_INJECT_MAX_TOKENS = 200

# ── Memory base corpus: SciQ ───────────────────────────────────────────────────
DATABASE_NAME = "sciq"
CORPUS_FILE = "sciq/train-00000-of-00001.parquet"

# ── Smoke test ─────────────────────────────────────────────────────────────────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(_ROOT, "data")
MMLU_FILE = os.path.join(_HERE, "data", "mmlu_n200.json")
RESULTS_DIR = os.path.join(_HERE, "results")
