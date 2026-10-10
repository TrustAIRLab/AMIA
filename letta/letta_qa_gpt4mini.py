"""
Letta QA memory-injection attack, GPT-4.1-mini (table:single_letta_results).

Runs every attack in configs/letta_gpt4mini_configs.py against the Letta agent
framework. Requires the `letta` package (separate environment -- see
requirements_letta.txt) and OPENAI_API_KEY in a `.env` at the repo root. Run
from the repo root:

    python letta/letta_qa_gpt4mini.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from configs import letta_gpt4mini_configs as C
import letta_core

if __name__ == "__main__":
    letta_core.run(C)
