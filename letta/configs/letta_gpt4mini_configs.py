"""
Config for the Letta QA memory-injection attack, GPT-4.1-mini backbone.

Produces the GPT column of table:single_letta_results (StrategyQA, N=200). The
only environment variable is the OpenAI key (OPENAI_API_KEY in a `.env` at the
repo root). Requires the `letta` package -- see requirements_letta.txt /
DATA_PREPARATION.md (separate environment from the main runners).
"""
import os

MODEL = "gpt-4.1-mini"
MODEL_ENDPOINT_TYPE = "openai"
MODEL_ENDPOINT = "https://api.openai.com/v1"
CONTEXT_WINDOW = 128000

N_QUERIES = 200
# PI + 3 Semi-I + 3 No-I + no_attack (baseline). table:single_letta_results
# reports PI and the BEST Semi-I / BEST No-I variant per backbone. For
# GPT-4.1-mini (N=50) the best per family are:
#   PI = pi (92.0) ; Semi-I = counter_argument (68.0) ; No-I = verification (48.0)
#   baseline no_attack = 18.0
# Provenance: letta_experiment/results/gpt4mini/<attack>/*.json + summary.jsonl.
ATTACKS = ["pi", "restate_elaboration", "counter_argument", "discredit_restate",
           "why_inquiry", "elaboration", "verification", "no_attack"]

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # letta/
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_HERE, "results", "gpt4mini")
