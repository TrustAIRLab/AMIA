"""
Config for the Letta QA memory-injection attack, Llama-3.1-70B backbone.

Produces the Llama column of table:single_letta_results (StrategyQA, N=200).
Llama is served by a local vLLM OpenAI-compatible server with tool calling
enabled (see DATA_PREPARATION.md). Requires the `letta` package -- see
requirements_letta.txt (separate environment from the main runners).
"""
import os

MODEL = "llm_models/meta-llama/Llama-3.1-70B-Instruct"   # the vLLM-served model id
MODEL_ENDPOINT_TYPE = "openai"                           # vLLM is OpenAI-compatible
MODEL_ENDPOINT = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
CONTEXT_WINDOW = 8192
# Sampling settings for the Llama agent (passed to LLMConfig).
LLM_CONFIG_EXTRA = {"temperature": 0.0, "max_tokens": 512}

N_QUERIES = 200
# table:single_letta_results reports PI and the BEST Semi-I / BEST No-I per
# backbone. For Llama-3.1-70B (N=200) the best per family are:
#   PI = pi (99.5) ; Semi-I = discredit_restate (91.0) ; No-I = why_inquiry (60.0)
#   baseline no_attack = 18.0
# (The best Semi-I/No-I variant differs from GPT's -- counter_argument/verification.)
# Provenance: letta_experiment/results/<attack>/*.json ; No-I = why_inquiry
# shards (8 x 25 = 200): results/why_inquiry/*shard*of8.json.
ATTACKS = ["pi", "restate_elaboration", "counter_argument", "discredit_restate",
           "why_inquiry", "elaboration", "verification", "no_attack"]

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_DIR = os.path.join(_ROOT, "data")
RESULTS_DIR = os.path.join(_HERE, "results", "llama")
