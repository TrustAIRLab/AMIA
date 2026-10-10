"""
Config for LLM-as-judge stealth detection (table:stealth_detection).

A memory-auditor judge scores 200 stored entries per (agent, category) and
reports the detection rate (benign entries give the false-positive rate). The
corpus is built ONCE from the single-agent *Llama* runs and scored by BOTH
judges (Llama-3.1-70B and GPT-4.1-mini), matching the paper.

Input = the release single-agent runs' per-entry `*_intermediate.json`. Run the
single-agent Llama runners (200-target subset) for the attacks below plus
`no_attack` first -- see DATA_PREPARATION.md.
"""
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # llm_judges/
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# Corpus source: single-agent Llama results (200-target subset).
SINGLE_AGENT_RESULTS = os.path.join(_ROOT, "single_agent", "results")

# The two judges. The paper reports the TARGETED detector under both.
# Llama judge needs a local vLLM OpenAI-compatible server; GPT judge needs
# OPENAI_API_KEY in a `.env` at the repo root. Comment one out to run just one.
JUDGES = [
    {"name": "llama", "model": os.environ.get("LLAMA_MODEL_NAME", "llama-3.1-70b"),
     "base_url": os.environ.get("VLLM_BASE", "http://localhost:8000/v1"), "api_key": "EMPTY"},
    {"name": "gpt4mini", "model": "gpt-4.1-mini", "base_url": None, "api_key": None},
]
DETECTOR = "targeted"           # "targeted" (reported) | "general"
MAX_WORKERS = 16

# Cells: (agent, family, variant_dir, field). Benign cells use is_benign=True.
# variant_dir is relative to SINGLE_AGENT_RESULTS (the 200-subset run tags).
CELLS = [
    ("QA", "PI", "qa_200/pi", "actual_inject", False),
    ("QA", "Semi-I", "qa_200/discredit_restate", "actual_inject", False),
    ("QA", "No-I", "qa_200/why_inquiry", "actual_inject", False),
    ("QA", "benign", "qa_200/no_attack", "victim_generation", True),
    ("Multi-QA", "PI", "multiqa_commonsenseqa_200/pi", "actual_inject", False),
    ("Multi-QA", "Semi-I", "multiqa_commonsenseqa_200/restate_elaboration", "actual_inject", False),
    ("Multi-QA", "No-I", "multiqa_commonsenseqa_200/why_inquiry", "actual_inject", False),
    ("Multi-QA", "benign", "multiqa_commonsenseqa_200/no_attack", "victim_generation", True),
    ("ReAct", "PI", "react_200/pi", "actual_inject", False),
    ("ReAct", "Semi-I", "react_200/fake_search_restate_elab", "actual_inject", False),
    ("ReAct", "No-I", "react_200/multiobs_verification", "actual_inject", False),
    ("ReAct", "benign", "react_200/no_attack", "final_action", True),
]

# Smoke test: LIMIT > 0 scores only the first LIMIT entries per cell -> results_smoke/.
LIMIT = 0

RESULTS_DIR = os.path.join(_HERE, "results", "stealth")
