"""
AMIA vs MINJA on MMLU (N=200), GPT-4.1-mini.

The MMLU agent answers a multiple-choice question using a flat key-value DPR
memory with a SciQ base corpus (the same memory as the Multi-QA agent). For each
attack the attacker injects one (or, for MINJA, five) poisoned memory entries,
then a clean victim query is answered; ASR-A counts victim answers whose first
letter is the attacker's target option.

Attacks (all config-driven; one invocation runs all of C.ATTACKS):
  pi      -- an "updated reference" override that outputs the target verbatim (1x)
  semi_i  -- a restate-elaboration presupposition (1x)
  no_i    -- a why-inquiry presupposition (1x)
  minja   -- MINJA's 5 progressively-shortened rounds; best-of-5 ASR-A reported (5x)

All knobs live in `configs/mmlu_gpt4mini_configs.py`; OpenAI key in a `.env` at
the repo root. Requires the SciQ base corpus -- see DATA_PREPARATION.md. Run:

    python mmlu/mmlu_gpt4mini.py
"""
import os
import sys
import json
import threading
import concurrent.futures

from rouge import Rouge
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import mmlu_gpt4mini_configs as C
from amia.multiqa_database import MQADataBase
from amia.presupposition import STRATEGIES
from note_generator_mmlu import generate_minja_notes_mcq

load_dotenv(os.path.join(_ROOT, ".env"))
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_lock = threading.Lock()
_tokens = {"in": 0, "out": 0}


def gpt(prompt, max_tokens=128):
    r = client.chat.completions.create(
        model=C.MODEL, seed=C.SEED, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    if r.usage:
        with _lock:
            _tokens["in"] += r.usage.prompt_tokens
            _tokens["out"] += r.usage.completion_tokens
    return r.choices[0].message.content


def generate(prompts, max_tokens):
    out = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=C.MAX_WORKERS) as ex:
        futures = {ex.submit(gpt, p, max_tokens): i for i, p in enumerate(prompts)}
        for fut in concurrent.futures.as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


from mmlu_core import run_all  # noqa: E402
run_all(C, generate, Rouge(), MQADataBase, STRATEGIES, generate_minja_notes_mcq, _tokens)
