"""
A-MEM ReAct memory-injection attack, GPT-4.1-mini backbone (paper Table 2,
A-MEM/ReAct).

The ReAct agent is GPT-4.1-mini; Search[] retrieves from an A-MEM store behind a
WikiEnv adapter. The attacker bulk-writes malicious Finish[] payloads through
A-MEM's add_note() (A-MEM auto-generates keywords/context/tags with its internal
LLM and stores them in ChromaDB). The vendored `agentic_memory` package sits next
to this file. Run from the repo root:

    python production_memory/AMem/amem_react_gpt4mini.py

Metrics at ROUGE-L recall >= C.ROUGE_TAU = 0.3 (ASR-I/ASR-R); ASR-A = the agent's
Finish[] answer matches the attacker's flipped target.

Note: Unlike MemoryOS, which preloads the benign base corpus to mirror the
flat-KV store, this memory holds the injected conversation turns -- the base
corpus is isolated paragraphs with no query context and does not form a
user-assistant conversation, the unit these conversational-memory frameworks
ingest; preloading it would break the memory's native-format consistency.
"""
import os
import sys

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import amem_react_gpt4mini_configs as C

import nltk
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

from amia import react_memory_gpt as driver
from amem_wikienv import AMemWikiEnv

load_dotenv(os.path.join(_ROOT, ".env"))

RUN_TAG = "react_mem_amem"


def env_factory(attack):
    return AMemWikiEnv(
        knn=C.KNN, memory_type=C.MEMORY_TYPE,
        amem_llm_model=C.MODEL, amem_embed_model=C.AMEM_EMBED_MODEL,
        amem_evo_threshold=C.AMEM_EVO_THRESHOLD,
        amem_api_key=os.environ.get("OPENAI_API_KEY"),
    )


driver.run(C, env_factory, backend="amem", run_tag=RUN_TAG)
