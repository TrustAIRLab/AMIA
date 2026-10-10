"""
Letta QA memory-injection attack, Llama-3.1-70B (table:single_letta_results).

Runs every attack in configs/letta_llama_configs.py against the Letta agent
framework, with Llama served by a local vLLM OpenAI-compatible server (tool
calling enabled). Requires the `letta` package (separate environment -- see
requirements_letta.txt). Run from the repo root:

    python letta/letta_qa_llama.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from configs import letta_llama_configs as C
import letta_core

if __name__ == "__main__":
    letta_core.run(C)
