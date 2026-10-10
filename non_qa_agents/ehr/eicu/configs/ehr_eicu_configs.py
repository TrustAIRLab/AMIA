"""
Config for the EHRAgent / eICU medication-substitution memory-injection attack
(paper: clinical non-QA agent; AMIA vs MINJA, N=270 = 9 drug pairs x 30 tests).

EHRAgent answers eICU medication queries by generating + running code; the
attacker injects redirect notes so poisoned long-term memory entries later steer
benign queries to the attacker's target drug. The attack driver subprocesses the
vendored eICU-aware EHRAgent app (`ehragent_eicu/`).

Plain constants. BACKEND selects the LLM: "openai" (GPT via OPENAI_API_KEY in a
`.env` at the repo root) or "vllm" (local Llama at VLLM_BASE). Requires the eICU
database and autogen -- see DATA_PREPARATION.md.
"""
import os

# ── Backbone ────────────────────────────────────────────────────────────────
BACKEND = "openai"              # "openai" (GPT) | "vllm" (local Llama server)
MODEL = "gpt-4.1-mini"          # OpenAI model name, or the vLLM-served Llama name
VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")

# ── Attack / run parameters (paper defaults) ────────────────────────────────
N_PAIRS = 9                     # MINJA's 9 victim/target drug pairs
N_BENIGN_DISTRACTORS = 50       # benign queries used to precompute base memory
N_INJECT_MINJA = 15            # MINJA inject seeds per pair (x PSS modes)
N_INJECT_OURS = 30             # our inject templates per pair (AMIA budget)
N_TEST = 30                     # benign test queries per pair (ASR-A denom)
VARIANTS = ["minja_pss", "pi", "semi_i", "no_i"]
SEED = 42
MAX_WORKERS = 12

# ── Smoke test: LIMIT > 0 -> 3 pairs, few tests/benign (checks the pipeline) ──
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # non_qa_agents/ehr/eicu/
EHRAGENT_EICU_DIR = os.path.join(_HERE, "ehragent_eicu")   # vendored app; DB goes here (see DATA_PREPARATION.md)
DRUG_ONLY_FILE = os.path.join(_HERE, "drug_only.json")     # eICU query templates (vendored)
RESULTS_DIR = os.path.join(_HERE, "results")
