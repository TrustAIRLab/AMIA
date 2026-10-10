"""
Config for the MINJA baseline (iterative attack) on the single-agent Multi-QA
(CommonsenseQA MCQ) agent, Llama-3.1-70B (local vLLM). MINJA column of the
Multi-QA head-to-head rows. AMIA side: ../single_agent/multiqa_llama.py.
"""
import os

MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

# ── Data: CommonsenseQA questions + SciQ base memory ───────────────────────────
DATASET = "commonsenseqa"                      # question set (MCQ pkls)
# MINJA's base memory corpus is SciQ (the MQADataBase default used by the paper's
# MINJA Multi-QA run); AMIA uses the CommonsenseQA base, so the two sides differ.
DATABASE_NAME = "sciq"
CORPUS_FILE = "sciq/train-00000-of-00001.parquet"
# Paper uses the full CommonsenseQA target set (N=986); "200" runs the shared
# mqa_indices subset (apples-to-apples with GPT).
N_SAMPLE = "full"
USE_FULL = N_SAMPLE.lower() == "full"

# ── Attack / run parameters ────────────────────────────────────────────────────
N_ROUNDS = 5
SEED = 42
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3

# ── Smoke test ─────────────────────────────────────────────────────────────────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
