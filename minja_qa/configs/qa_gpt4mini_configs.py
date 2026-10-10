"""
Config for the MINJA baseline (iterative attack) on the single-agent QA agent,
GPT-4.1-mini backbone. Produces the MINJA column of the QA head-to-head table
(answer-based and input-key memory).

MINJA injects a progressively-shortened reasoning note over 5 rounds into one
shared flat key-value memory bank; ASR-A is reported as the best of the 5 rounds.
The AMIA side of the head-to-head is produced by ../single_agent/qa_gpt4mini.py.

Plain constants. The only environment variable is the OpenAI key, read from a
`.env` (OPENAI_API_KEY=...) at the repo root.
"""
import os

# ── Model (OpenAI API) ─────────────────────────────────────────────────────────
MODEL = "gpt-4.1-mini"
MAX_WORKERS = 32
INPUT_PRICE = 0.40              # $ per 1M tokens (cost report only)
OUTPUT_PRICE = 1.60

# ── Attack / run parameters ────────────────────────────────────────────────────
N_TARGETS = 200                 # GPT uses the 200 sampled StrategyQA targets
N_ROUNDS = 5                    # MINJA's 5 progressively-shortened rounds
SEED = 42

# Memory keying: "none" -> answer-based (key on the response); "input_key" ->
# question-based (key on the attacker query q_adv). These are the two QA rows.
ABLATION = "none"
assert ABLATION in {"none", "input_key"}
MEMORY_TYPE = "question_based" if ABLATION == "input_key" else "answer_based"
KNN = 1
ROUGE_TAU = 0.3                 # ASR-I / ASR-R threshold (diagnostics)

# ── Smoke test: LIMIT > 0 runs only the first LIMIT targets -> results_smoke/ ───
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_ROOT, "minja_qa", "results")
