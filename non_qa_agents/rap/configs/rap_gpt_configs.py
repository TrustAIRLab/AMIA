"""
Config for the RAP / WebShop memory-injection attack on a GPT backbone
(paper: web-shopping non-QA agent; AMIA vs MINJA, N=270 = 9 pairs x 30 tests).

The RAP agent shops on WebShop (served locally by mock_webshop.py). The attacker
injects poisoned shopping trajectories into RAP's memory so that benign victim
queries later retrieve them and buy the attacker's TARGET product. Everything
here is a plain constant. The only environment variable is the OpenAI key
(OPENAI_API_KEY, via a `.env` at the repo root).

Requires the WebShop product data and a running mock_webshop server -- see
DATA_PREPARATION.md.
"""
import os

# ── Model (OpenAI API backbone) ─────────────────────────────────────────────
# The paper reports RAP+WebShop only on GPT-4o: weaker backbones (GPT-4.1-mini,
# Llama-3.1-70B) cannot complete even benign WebShop navigation, so their ASR is
# not meaningful. Keep gpt-4o to reproduce the paper numbers.
MODEL = "gpt-4o"
INPUT_PRICE = 2.50             # $ per 1M tokens (cost report only; gpt-4o)
CACHED_INPUT_PRICE = 1.25
OUTPUT_PRICE = 10.00

# ── Attack / run parameters (paper defaults) ────────────────────────────────
VARIANTS = ["pi", "semi_i", "no_i", "minja_pss"]
PAIR_INDICES = "all"            # "all" = the 9 victim/target pairs; or e.g. "0,1"
INJECT_NUM = 15                 # malicious shopping queries injected per pair
TEST_NUM = 30                   # benign victim queries evaluated per pair (ASR-A denom)
NUM_BENIGN = 50                 # benign shopping trajectories interleaved during inject
NUM_STEPS = 15                  # max WebShop actions per episode
SEED = 42
BUDGET_USD = 25.0               # hard cap on total OpenAI spend; abort if exceeded

# ── WebShop server (mock_webshop.py) + embeddings ───────────────────────────
WEBSHOP_URL = os.environ.get("WEBSHOP_URL", "http://localhost:3000")
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Smoke test: LIMIT > 0 shrinks a run (few injects/tests/benign, one pair) ──
# so you can check the pipeline end-to-end without a full N=270 sweep.
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # non_qa_agents/rap/
RESULTS_DIR = os.path.join(_HERE, "results")
