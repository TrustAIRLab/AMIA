"""
Mem0 ReAct memory-injection attack, GPT-4.1-mini backbone (paper Table 2,
Mem0/ReAct).

The ReAct agent (Thought/Action/Observation over a StrategyQA Wikipedia env) is
GPT-4.1-mini; Search[] retrieves from a Mem0 store behind a WikiEnv adapter. The
attacker bulk-writes malicious Finish[] payloads through Mem0's add() (Mem0's
internal LLM extracts facts into qdrant). All knobs live in
`configs/mem0_react_gpt4mini_configs.py`; the OpenAI key is read from a `.env` at
the repo root. Run from the repo root:

    python production_memory/Mem0/mem0_react_gpt4mini.py

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

from configs import mem0_react_gpt4mini_configs as C
from amia import react_memory_gpt as driver
from mem0_wikienv import Mem0WikiEnv

load_dotenv(os.path.join(_ROOT, ".env"))

RUN_TAG = "react_mem_mem0"
RESULTS_ROOT = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR


def env_factory(attack):
    return Mem0WikiEnv(
        knn=C.KNN, memory_type=C.MEMORY_TYPE,
        mem0_data_dir=os.path.join(RESULTS_ROOT, RUN_TAG, attack, "mem0_store"),
        mem0_collection_name=f"react_amia_{attack}",
        mem0_llm_model=C.MEM0_MODEL, mem0_embedder=C.MEM0_EMBEDDER,
    )


driver.run(C, env_factory, backend="mem0", run_tag=RUN_TAG)
