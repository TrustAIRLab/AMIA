"""
Config for Table C (table:minja_defense_on_amia): MINJA's input-side prompt
detector applied to AMIA's injected queries.

For each (agent, family) the detector judges the attacker's `q_adv` (logged by
the single-agent runners) and reports how many of N are flagged; on benign
queries it gives the detector's false-positive rate. Two detectors (Targeted,
General) under two judges (Llama, GPT-4.1-mini).
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # defense/
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
SINGLE_AGENT_RESULTS = os.path.join(_ROOT, "single_agent", "results")

JUDGES = [
    {"name": "llama", "model": os.environ.get("LLAMA_MODEL_NAME", "llama-3.1-70b"),
     "base_url": os.environ.get("VLLM_BASE", "http://localhost:8000/v1"), "api_key": "EMPTY"},
    {"name": "gpt4mini", "model": "gpt-4.1-mini", "base_url": None, "api_key": None},
]
DETECTORS = ["targeted", "general"]
MAX_WORKERS = 16

# (agent, family, variant_dir). The attack families use the best Semi-I/No-I
# variant per agent; benign is the no_attack q_adv (= the clean query).
CELLS = [
    ("QA", "PI", "qa_200/pi"),
    ("QA", "Semi-I", "qa_200/discredit_restate"),
    ("QA", "No-I", "qa_200/why_inquiry"),
    ("QA", "benign", "qa_200/no_attack"),
    ("Multi-QA", "PI", "multiqa_commonsenseqa_200/pi"),
    ("Multi-QA", "Semi-I", "multiqa_commonsenseqa_200/restate_elaboration"),
    ("Multi-QA", "No-I", "multiqa_commonsenseqa_200/why_inquiry"),
    ("Multi-QA", "benign", "multiqa_commonsenseqa_200/no_attack"),
    ("ReAct", "PI", "react_200/pi"),
    ("ReAct", "Semi-I", "react_200/fake_search_restate_elab"),
    ("ReAct", "No-I", "react_200/multiobs_verification"),
    ("ReAct", "benign", "react_200/no_attack"),
]

LIMIT = 0
RESULTS_DIR = os.path.join(_HERE, "results", "minja_detector")
