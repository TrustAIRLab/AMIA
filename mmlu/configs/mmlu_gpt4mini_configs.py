"""
Config for AMIA vs MINJA on MMLU (N=200), GPT-4.1-mini backbone.

One runner evaluates all four attacks (PI, Semi-I, No-I, MINJA) on the MMLU
multiple-choice agent, which uses the same flat key-value DPR memory as the
Multi-QA agent with a SciQ base corpus. PI/Semi-I/No-I are 1x cost (one inject
round); MINJA is 5x (five progressively-shortened rounds, best-of-5 reported).

Plain constants. The only environment variable is the OpenAI key, read from a
`.env` at the repo root. Requires the SciQ base corpus -- see DATA_PREPARATION.md.
"""
import os

MODEL = "gpt-4.1-mini"
MAX_WORKERS = 20
INPUT_PRICE = 0.40
OUTPUT_PRICE = 1.60

# ── Attack / run parameters ────────────────────────────────────────────────────
ATTACKS = ["pi", "semi_i", "no_i", "minja"]
N_ROUNDS = 5                    # MINJA rounds (best-of-5 reported)
SEED = 42
MEMORY_TYPE = "question_based"  # key memory on the attacker query (the MMLU setting)
KNN = 1
ROUGE_TAU = 0.3

# ── Memory base corpus: SciQ (reader-provided parquet with a `support` column) ─
DATABASE_NAME = "sciq"
CORPUS_FILE = "sciq/train-00000-of-00001.parquet"   # relative to DATA_DIR

# ── Smoke test: LIMIT > 0 -> first LIMIT MMLU questions -> results_smoke/ ───────
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # mmlu/
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(_ROOT, "data")                 # shared corpora (SciQ lives here)
MMLU_FILE = os.path.join(_HERE, "data", "mmlu_n200.json")
RESULTS_DIR = os.path.join(_HERE, "results")
