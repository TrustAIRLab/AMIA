"""
Config for the MINJA baseline (iterative attack) on the single-agent QA agent,
Llama-3.1-70B backbone (local vLLM). Produces the MINJA column of the QA
head-to-head table (answer-based and input-key memory).

The AMIA side is produced by ../single_agent/qa_llama.py. No environment
variables (Llama is served locally).
"""
import os

# ── Model: local vLLM ──────────────────────────────────────────────────────────
MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

# ── Data: StrategyQA ───────────────────────────────────────────────────────────
# Paper uses the full set (N=2270); "200" runs the shared GPT subset.
N_SAMPLE = "full"
USE_FULL = N_SAMPLE.lower() == "full"

# ── Attack / run parameters ────────────────────────────────────────────────────
N_ROUNDS = 5
SEED = 6

# "none" -> answer-based memory; "input_key" -> question-based (key on q_adv).
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3

# ── Smoke test: LIMIT > 0 runs only the first LIMIT targets -> results_smoke/ ───
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
