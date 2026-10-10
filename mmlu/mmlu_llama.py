"""
AMIA vs MINJA on MMLU (N=200), Llama-3.1-70B (local vLLM).

Same four attacks and metrics as mmlu_gpt4mini.py, with a local vLLM backbone.
All knobs live in `configs/mmlu_llama_configs.py`. Requires the SciQ base corpus
and local Llama weights -- see DATA_PREPARATION.md. Run from the repo root:

    python mmlu/mmlu_llama.py
"""
import os
import sys

from rouge import Rouge
from vllm import LLM, SamplingParams

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import mmlu_llama_configs as C
from amia.multiqa_database import MQADataBase
from amia.presupposition import STRATEGIES
from note_generator_mmlu import generate_minja_notes_mcq

llm = LLM(model=C.MODEL_PATH, tensor_parallel_size=C.TENSOR_PARALLEL_SIZE,
          gpu_memory_utilization=C.GPU_MEMORY_UTILIZATION, dtype="bfloat16",
          seed=C.SEED, enable_prefix_caching=True, max_model_len=C.MAX_MODEL_LEN)


def generate(prompts, max_tokens):
    params = SamplingParams(seed=C.SEED, temperature=0, max_tokens=max_tokens)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


from mmlu_core import run_all  # noqa: E402
run_all(C, generate, Rouge(), MQADataBase, STRATEGIES, generate_minja_notes_mcq)
