"""
MemoryOS ReAct memory-injection attack, Llama-3.1-70B backbone (paper Table 2,
MemoryOS/ReAct).

The ReAct agent is offline vLLM Llama, which is also installed as MemoryOS's
native topic-summary backend; embeddings are local all-MiniLM. No OpenAI key.
Search[] retrieves from MemoryOS behind a WikiEnv adapter (MemoryOSAdapter, which
preloads the StrategyQA paragraph corpus as the benign base memory); the attacker
bulk-writes malicious Finish[] payloads. Run from the repo root with the vLLM
weights available:

    python production_memory/MemoryOS/memoryos_react_llama.py

Metrics at ROUGE-L recall >= C.ROUGE_TAU = 0.3 (ASR-I/ASR-R); ASR-A = the agent's
Finish[] answer matches the attacker's flipped target.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import memoryos_react_llama_configs as C
from amia import react_memory_llama as driver
from memoryos_wikienv import MemoryOSWikiEnv
from vllm import LLM, SamplingParams

RUN_TAG = ("react_mem_memoryos_key_qadv_pre_native_inj_bulk_stm_history"
           + ("_stmbenign" if C.MEMORYOS_STM_BENIGN else "")
           + (f"_knn{C.MEMORYOS_KNN}" if C.MEMORYOS_KNN != 1 else ""))
RESULTS_ROOT = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR
BASE_DB = os.path.join(C.DATA_DIR, C.CORPUS_FILE)
TEMPLATE_DIR = os.path.join(RESULTS_ROOT, RUN_TAG, "_preloaded_template")

llm = LLM(model=C.MODEL_PATH, tensor_parallel_size=C.TENSOR_PARALLEL_SIZE,
          gpu_memory_utilization=C.GPU_MEMORY_UTILIZATION)


def gen(prompts, max_tokens, stop=None):
    params = SamplingParams(seed=6, temperature=0, max_tokens=max_tokens, stop=stop)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


# Install Llama as MemoryOS's native topic-summary backend (PRELOAD=native).
if C.MEMORYOS_PRELOAD == "native":
    from memoryos_adapter import install_llm_summary_backend
    _tok = llm.get_tokenizer()

    def _llama_single(messages, max_tokens=512):
        prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        p = SamplingParams(seed=42, temperature=0, max_tokens=max_tokens)
        return llm.generate([prompt], p, use_tqdm=False)[0].outputs[0].text

    def _llama_batch(messages_list, max_tokens=512):
        prompts = [_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list]
        p = SamplingParams(seed=42, temperature=0, max_tokens=max_tokens)
        return [o.outputs[0].text for o in llm.generate(prompts, p, use_tqdm=False)]

    install_llm_summary_backend(_llama_single, _llama_batch)
    print("MemoryOS: installed Llama as native summary backend")


def env_factory(attack):
    return MemoryOSWikiEnv(
        knn=C.KNN, memory_type=C.MEMORY_TYPE, base_db_path=BASE_DB,
        data_storage_path=os.path.join(RESULTS_ROOT, RUN_TAG, attack, "memoryos_data"),
        template_data_path=TEMPLATE_DIR, stm_mode=C.MEMORYOS_STM_MODE,
        embedding_model_name=C.EMBEDDING_MODEL, embedding_device=C.EMBEDDING_DEVICE,
        memoryos_knn=C.MEMORYOS_KNN,
    )


driver.run(C, gen, env_factory, backend="memoryos", run_tag=RUN_TAG,
           stm_benign=C.MEMORYOS_STM_BENIGN)
