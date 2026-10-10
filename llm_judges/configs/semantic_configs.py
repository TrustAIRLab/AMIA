"""
Config for semantic validation of ASR-I / ASR-R (table:semantic_validation).

A GPT-4.1-mini entailment judge decides whether each stored entry (ASR-I) /
retrieved passage (ASR-R) communicates the attacker's target claim, as a
semantic complement to the ROUGE-L (tau=0.3) criterion. 200 entries per
(agent, family), 3 agents x 3 families = N=1800 each for I and R. GPT judge only.

Input = the release single-agent *GPT-4.1-mini* runs' `*_intermediate.json`
(payloads) and `*_final.json` (ASR-A). Run those first -- see DATA_PREPARATION.md.
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # llm_judges/
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

SINGLE_AGENT_RESULTS = os.path.join(_ROOT, "single_agent", "results")

# GPT-4.1-mini entailment judge (OPENAI_API_KEY in a `.env` at the repo root).
JUDGE = {"model": "gpt-4.1-mini", "base_url": None, "api_key": None}
MAX_WORKERS = 16
ROUGE_TAU = 0.3

# Cells: (agent, family, variant_dir) relative to SINGLE_AGENT_RESULTS. The
# family -> attack-variant mapping is the best-per-family used by the paper.
CELLS = [
    ("QA", "PI", "qa/pi"),
    ("QA", "Semi-I", "qa/discredit_restate"),
    ("QA", "No-I", "qa/why_inquiry"),
    ("Multi-QA", "PI", "multiqa_commonsenseqa/pi"),
    ("Multi-QA", "Semi-I", "multiqa_commonsenseqa/restate_elaboration"),
    ("Multi-QA", "No-I", "multiqa_commonsenseqa/why_inquiry"),
    ("ReAct", "PI", "react/pi"),
    ("ReAct", "Semi-I", "react/fake_search_restate_elab"),
    ("ReAct", "No-I", "react/multiobs_verification"),
]

# Smoke test: LIMIT > 0 judges only the first LIMIT entries per cell -> results_smoke/.
LIMIT = 0

RESULTS_DIR = os.path.join(_HERE, "results", "semantic")
