"""
Mem0 ReAct memory-injection attack, Llama-3.1-70B backbone (paper Table 2,
Mem0/ReAct).

The ReAct agent is local vLLM Llama; Mem0's internal fact-extractor LLM and
embedder stay on OpenAI (key via `.env`). Search[] retrieves from the Mem0 store
behind a WikiEnv adapter; the attacker bulk-writes malicious Finish[] payloads.
Run from the repo root with the vLLM weights available:

    python production_memory/Mem0/mem0_react_llama.py

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

from configs import mem0_react_llama_configs as C
from amia import react_memory_llama as driver
from mem0_wikienv import Mem0WikiEnv
from vllm import LLM, SamplingParams

load_dotenv(os.path.join(_ROOT, ".env"))

RUN_TAG = "react_mem_mem0"
RESULTS_ROOT = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR

llm = LLM(model=C.MODEL_PATH, tensor_parallel_size=C.TENSOR_PARALLEL_SIZE,
          gpu_memory_utilization=C.GPU_MEMORY_UTILIZATION)


def gen(prompts, max_tokens, stop=None):
    params = SamplingParams(seed=6, temperature=0, max_tokens=max_tokens, stop=stop)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


def env_factory(attack):
    return Mem0WikiEnv(
        knn=C.KNN, memory_type=C.MEMORY_TYPE,
        mem0_data_dir=os.path.join(RESULTS_ROOT, RUN_TAG, attack, "mem0_store"),
        mem0_collection_name=f"react_amia_{attack}",
        mem0_llm_model=C.MEM0_MODEL, mem0_embedder=C.MEM0_EMBEDDER,
    )


driver.run(C, gen, env_factory, backend="mem0", run_tag=RUN_TAG)
