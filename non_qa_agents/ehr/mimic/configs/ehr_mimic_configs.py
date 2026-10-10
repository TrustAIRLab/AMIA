"""
Config for the EHRAgent / MIMIC-III patient-ID-substitution memory-injection
attack (paper: clinical non-QA agent; AMIA vs MINJA, Patient ID redirection).

EHRAgent answers MIMIC-III clinical queries by generating + running code; the
attacker injects redirect notes so poisoned long-term memory entries later steer
benign queries from the victim patient ID to the attacker's target patient ID.
The driver subprocesses the vendored MIMIC-aware EHRAgent app (`ehragent/`).

Plain constants. BACKEND selects the LLM: "openai" (GPT via OPENAI_API_KEY in a
`.env` at the repo root) or "vllm" (local Llama at VLLM_BASE). The vendored app
routes by model-name prefix, so MODEL must be a real OpenAI name for "openai".
Requires the MIMIC-III data files and autogen -- see DATA_PREPARATION.md.
"""
import os

# ── Backbone ────────────────────────────────────────────────────────────────
BACKEND = "openai"              # "openai" (GPT) | "vllm" (local Llama server)
MODEL = "gpt-4.1-mini"          # OpenAI model name, or the vLLM-served Llama name
VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")

# ── Attack / run parameters (paper defaults) ────────────────────────────────
N_PAIRS = 9                     # MINJA's 9 victim/target patient pairs
N_BENIGN_DISTRACTORS = 50       # benign queries used to precompute base memory
N_INJECT_MINJA = 15             # MINJA inject seeds per pair (PSS modes cycled)
N_INJECT_OURS = 10              # our inject seeds per pair
N_TEST = 30                     # benign test queries per pair (ASR-A denom)
VARIANTS = ["minja_pss", "pi", "semi_i", "no_i"]
SEED = 42
MAX_WORKERS = 12

# ── Smoke test: LIMIT > 0 -> 3 pairs, few tests/benign (checks the pipeline) ──
LIMIT = 0

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # non_qa_agents/ehr/mimic/
EHRAGENT_DIR = os.path.join(_HERE, "ehragent")                          # vendored app
MIMIC_DATA_DIR = os.path.join(_HERE, "ehrsql-ehragent", "mimic_iii")    # reader-provided MIMIC files
DATASET_PATIENT_ONLY = os.path.join(MIMIC_DATA_DIR, "patient_only.json")  # query templates
PATIENTS_CSV = os.path.join(MIMIC_DATA_DIR, "PATIENTS.csv")               # patient-ID pool
# (The app reads mimic_iii.db from MIMIC_DATA_DIR via a path relative to ehragent/.)
RESULTS_DIR = os.path.join(_HERE, "results")
