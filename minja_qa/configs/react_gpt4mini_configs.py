"""
Config for the MINJA baseline (iterative attack) on the single-agent ReAct agent,
GPT-4.1-mini. Produces the MINJA column of the ReAct head-to-head rows
(answer-based and input-key memory).

The AMIA side is ../single_agent/react_gpt4mini.py. The only environment
variable is the OpenAI key, read from a `.env` at the repo root.
"""
import os

MODEL = "gpt-4.1-mini"
MAX_WORKERS = 32
INPUT_PRICE = 0.40
OUTPUT_PRICE = 1.60

# ── Attack / run parameters ────────────────────────────────────────────────────
N_TARGETS = 200                 # GPT uses the 200 sampled ReAct targets
N_ROUNDS = 5                    # MINJA's 5 progressively-shortened rounds
MAX_VICTIM_ROUNDS = 4           # max Thought/Action/Observation steps per victim rollout
SEED = 42
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3
PROMPTS_FILE = "react_prompts.json"   # the ReAct few-shot examples (key "sqa_react")

# ── Smoke test: LIMIT > 0 -> first LIMIT targets -> results_smoke/ ─────────────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
