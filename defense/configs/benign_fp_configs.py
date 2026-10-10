"""
Config for Table B (table:benign_fp_consistency): the false-positive rate of the
Consistency defense on benign (no-attack) memory writes.

Input = the single-agent `no_attack` runs' stored benign responses. Scored by the
same Consistency check used as a defense (amia/defenses.CONSISTENCY_TEMPLATE); a
benign response judged "inconsistent" is a false positive. Reported for the
Llama and GPT-4.1-mini judges.
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
MAX_WORKERS = 16

# (agent, no_attack intermediate dir, question_field, response_field). ReAct's
# benign response is its Finish action; entries that never reach Finish are empty
# and skipped (so the ReAct denominator is ~42/200).
CELLS = [
    ("QA", "qa_200/no_attack", "eval_question", "victim_generation"),
    ("Multi-QA", "multiqa_commonsenseqa_200/no_attack", "eval_question", "victim_generation"),
    ("ReAct", "react_200/no_attack", "eval_question", "final_action"),
]

LIMIT = 0
RESULTS_DIR = os.path.join(_HERE, "results", "benign_fp")
