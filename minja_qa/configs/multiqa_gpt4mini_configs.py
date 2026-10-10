"""
Config for the MINJA baseline (iterative attack) on the single-agent Multi-QA
(CommonsenseQA MCQ) agent, GPT-4.1-mini. Produces the MINJA column of the
Multi-QA head-to-head rows (answer-based and input-key memory).

The AMIA side is ../single_agent/multiqa_gpt4mini.py. The only environment
variable is the OpenAI key, read from a `.env` at the repo root.
"""
import os

MODEL = "gpt-4.1-mini"
MAX_WORKERS = 32
INPUT_PRICE = 0.40
OUTPUT_PRICE = 1.60

# ── Data: CommonsenseQA questions + SciQ base memory ───────────────────────────
DATASET = "commonsenseqa"                      # question set (MCQ pkls)
# MINJA's base memory corpus is SciQ -- this matches the paper's MINJA Multi-QA
# run (which used the MQADataBase default base). The AMIA side (single_agent)
# uses the CommonsenseQA base, so the two head-to-head sides differ by design.
DATABASE_NAME = "sciq"
CORPUS_FILE = "sciq/train-00000-of-00001.parquet"   # base corpus (relative to DATA_DIR)
# GPT uses the shared sampled MCQ targets (mqa_indices); a few indices fall
# outside the local corpus, so the effective count is ~195.
N_TARGETS = 200

# ── Attack / run parameters ────────────────────────────────────────────────────
N_ROUNDS = 5
SEED = 42
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3

# ── Smoke test: LIMIT > 0 -> first LIMIT targets -> results_smoke/ ──────────────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
