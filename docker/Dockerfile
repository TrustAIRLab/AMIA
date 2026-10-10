# AMIA — runtime image for the Llama-3.1-70B and GPT-4.1-mini runners.
#
# Three isolated Python 3.11 virtualenvs (their pins conflict, keep them separate):
#   /opt/venvs/amia         all runners except Letta; Llama runs in-process with vLLM 0.5.5
#   /opt/venvs/letta        Letta 0.10.0 client (letta/)
#   /opt/venvs/vllm_server  vLLM 0.8.5 OpenAI-compatible server with tool calling, used as the
#                           Llama backend for Letta and for the Llama judges in defense/ and llm_judges/
# Each env installs its requirements file (requirements.txt, letta/requirements_letta.txt,
# docker/requirements-vllm-server.txt) constrained by the matching docker/requirements-lock-*.txt.
#
# Build (context = repository root):
#   docker build -f docker/Dockerfile -t amia .
#
# Run (4x 80GB GPUs for Llama-3.1-70B with tensor parallelism 4; NVIDIA driver >= 535):
#   docker run --gpus all --ipc=host --shm-size=64g -it \
#     -v $PWD:/workspace \
#     -v /path/to/Llama-3.1-70B-Instruct:/workspace/llm_models/meta-llama/Llama-3.1-70B-Instruct:ro \
#     -e OPENAI_API_KEY=sk-... \
#     amia
# Mounting the weights at llm_models/meta-llama/Llama-3.1-70B-Instruct matches the default
# MODEL_PATH in every *_llama_configs.py. OPENAI_API_KEY is only needed by the GPT-4.1-mini
# runners, by Mem0 (its internal LLM/embedder) and by the GPT judges.
#
# Inside the container (working directory = repository root):
#   python single_agent/qa_llama.py                        # any runner, see each folder's DATA_PREPARATION.md
#   # Llama server for Letta, A-MEM (QA / Multi-QA) and the Llama judges (one server, all model ids):
#   /opt/venvs/vllm_server/bin/python -m vllm.entrypoints.openai.api_server \
#     --model llm_models/meta-llama/Llama-3.1-70B-Instruct \
#     --served-model-name llm_models/meta-llama/Llama-3.1-70B-Instruct llama-3.1-70B llama-3.1-70b \
#     --tensor-parallel-size 4 --port 8000 --max-model-len 8192 \
#     --tool-call-parser llama3_json --enable-auto-tool-choice &
#   /opt/venvs/letta/bin/python letta/letta_qa_llama.py
#   python defense/minja_detector_on_amia.py
# The in-process Llama runners and the server both need all four GPUs; run them one at a time.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    AMIA_DEVICE=cuda \
    PYTORCH_NVML_BASED_CUDA_CHECK=1

# build-essential: triton JIT-compiles its CUDA helpers the first time vLLM starts.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN for e in amia letta vllm_server; do \
      python3.11 -m venv /opt/venvs/$e && /opt/venvs/$e/bin/pip install -U pip "setuptools<81" wheel; \
    done

COPY requirements.txt /tmp/req/requirements-main.txt
COPY letta/requirements_letta.txt /tmp/req/requirements-letta.txt
COPY docker/requirements-vllm-server.txt docker/requirements-lock-main.txt \
     docker/requirements-lock-letta.txt docker/requirements-lock-vllm-server.txt /tmp/req/
COPY third_party/pyairports_stub /tmp/pyairports_stub

# outlines==0.0.43 (pinned by vllm==0.5.5) needs `pyairports`; install the bundled package first.
RUN /opt/venvs/amia/bin/pip install --no-deps /tmp/pyairports_stub \
 && /opt/venvs/amia/bin/pip install -r /tmp/req/requirements-main.txt -c /tmp/req/requirements-lock-main.txt \
 && /opt/venvs/amia/bin/pip check

RUN /opt/venvs/letta/bin/pip install -r /tmp/req/requirements-letta.txt -c /tmp/req/requirements-lock-letta.txt \
 && /opt/venvs/letta/bin/pip check

RUN /opt/venvs/vllm_server/bin/pip install -r /tmp/req/requirements-vllm-server.txt -c /tmp/req/requirements-lock-vllm-server.txt \
 && /opt/venvs/vllm_server/bin/pip check

# Small Hugging Face models the runners load by hub id: the DPR memory retriever and the
# MiniLM embedders (MemoryOS / A-MEM / Letta). Cached at build time so runs work offline.
RUN /opt/venvs/amia/bin/python -c "\
from transformers import AutoTokenizer, DPRContextEncoder; \
AutoTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base'); \
DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')" \
 && /opt/venvs/letta/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')" \
 && chmod -R a+rX $HF_HOME

ENV PATH=/opt/venvs/amia/bin:$PATH
WORKDIR /workspace
CMD ["bash"]
