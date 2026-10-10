"""
Config for the MINJA baseline (iterative attack) on the single-agent ReAct agent,
Llama-3.1-70B (local vLLM). MINJA column of the ReAct head-to-head rows.
AMIA side: ../single_agent/react_llama.py.
"""
import os

MODEL_PATH = "llm_models/meta-llama/Llama-3.1-70B-Instruct"
MODEL_NAME = "llama3.1-70B"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.85

# ── Data: StrategyQA ReAct ─────────────────────────────────────────────────────
# Paper uses the full set (N=2270); "200" runs the shared react_indices subset.
N_SAMPLE = "full"
USE_FULL = N_SAMPLE.lower() == "full"

# ── Attack / run parameters ────────────────────────────────────────────────────
N_ROUNDS = 5
MAX_VICTIM_ROUNDS = 4
SEED = 6
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3
PROMPTS_FILE = "react_prompts.json"

# ── Smoke test ─────────────────────────────────────────────────────────────────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
