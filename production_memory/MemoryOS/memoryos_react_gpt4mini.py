"""
MemoryOS ReAct memory-injection attack, GPT-4.1-mini backbone (paper Table 2,
MemoryOS/ReAct).

The ReAct agent is GPT-4.1-mini; Search[] retrieves from MemoryOS behind a WikiEnv
adapter (MemoryOSAdapter, which preloads the StrategyQA paragraph corpus as the
benign base memory). GPT is also installed as MemoryOS's native topic-summary LLM.
The attacker bulk-writes malicious Finish[] payloads. All knobs live in
`configs/memoryos_react_gpt4mini_configs.py`; the OpenAI key is read from a `.env`
at the repo root. Run from the repo root:

    python production_memory/MemoryOS/memoryos_react_gpt4mini.py

Metrics at ROUGE-L recall >= C.ROUGE_TAU = 0.3 (ASR-I/ASR-R); ASR-A = the agent's
Finish[] answer matches the attacker's flipped target.
"""
import os
import sys

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import memoryos_react_gpt4mini_configs as C
from amia import react_memory_gpt as driver
from memoryos_wikienv import MemoryOSWikiEnv

load_dotenv(os.path.join(_ROOT, ".env"))

RUN_TAG = ("react_mem_memoryos_key_qadv_pre_native_inj_bulk_stm_history"
           + ("_stmbenign" if C.MEMORYOS_STM_BENIGN else "")
           + (f"_knn{C.MEMORYOS_KNN}" if C.MEMORYOS_KNN != 1 else ""))
RESULTS_ROOT = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR
BASE_DB = os.path.join(C.DATA_DIR, C.CORPUS_FILE)
TEMPLATE_DIR = os.path.join(RESULTS_ROOT, RUN_TAG, "_preloaded_template")


def env_factory(attack):
    return MemoryOSWikiEnv(
        knn=C.KNN, memory_type=C.MEMORY_TYPE, base_db_path=BASE_DB,
        data_storage_path=os.path.join(RESULTS_ROOT, RUN_TAG, attack, "memoryos_data"),
        template_data_path=TEMPLATE_DIR, stm_mode=C.MEMORYOS_STM_MODE,
        embedding_model_name=C.EMBEDDING_MODEL, embedding_device=C.EMBEDDING_DEVICE,
        memoryos_knn=C.MEMORYOS_KNN,
    )


def on_ready(single, batch):
    if C.MEMORYOS_PRELOAD == "native":
        from memoryos_adapter import install_llm_summary_backend
        install_llm_summary_backend(single, batch)
        print("MemoryOS: installed GPT-4.1-mini as native summary backend")


driver.run(C, env_factory, backend="memoryos", run_tag=RUN_TAG,
           stm_benign=C.MEMORYOS_STM_BENIGN, on_ready=on_ready)
