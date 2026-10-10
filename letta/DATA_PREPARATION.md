# Letta — environment & run

Letta (MemGPT) is a production agent framework whose agent autonomously manages a
shared core-memory block via built-in tools (`core_memory_append/replace`). We
attack it exactly through the standard conversation API: the adversary only sends
chat messages, and the agent decides what to store. `table:single_letta_results`
is StrategyQA, N=200, PI / Semi-I / No-I vs a no-attack baseline.

## 1. Separate environment (important)
Letta's dependency tree conflicts with the main `requirements.txt`, so install it
in its OWN environment and run only `letta/` from there:
```
python -m venv .venv_letta
. .venv_letta/bin/activate
pip install -r non_prompt_injection/new_published/letta/requirements_letta.txt
```
The experiment used **letta 0.10.0** (recorded in the original
`letta_experiment/LETTA_EXPERIMENT.md`: "Letta v0.10.0 (pip install), SyncServer
with SQLite backend"). `letta_core.py` runs on 0.10.0 unmodified (its embedded
`SyncServer` still exposes the sync `create_agent` / `send_messages` /
`get_agent_memory`). Do NOT use the current letta (0.16.x+): it moved SyncServer
to async and dropped `send_messages`, so the scripts won't run. Pin exactly
`letta==0.10.0` (see requirements_letta.txt). On first run Letta creates
`~/.letta/sqlite.db` and loads the `sqlite-vec` extension.

## 2. Data (already shipped)
Reuses the StrategyQA files under `../data/` (`strategyqa_train.json`,
`contradiction_results.json`) — the same corpus the single-agent QA runner uses.
No download needed.

## 3. Backbone
- **GPT** (`letta_qa_gpt4mini.py`): `OPENAI_API_KEY` in a `.env` at the repo root.
- **Llama** (`letta_qa_llama.py`): a local vLLM OpenAI-compatible server with tool
  calling enabled:
  ```
  python -m vllm.entrypoints.openai.api_server \
    --model llm_models/meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --port 8000 --max-model-len 8192 \
    --tool-call-parser llama3_json --enable-auto-tool-choice
  ```
  Point `VLLM_BASE` at it if not `http://localhost:8000/v1`.

## 4. Run
From the repo root (each runs all configured attacks; `no_attack` = baseline):
```
python letta/letta_qa_gpt4mini.py
python letta/letta_qa_llama.py
```
Per attack it runs the 3-phase episode (baseline → inject `q_adv` → re-ask) on a
fresh agent and reports ASR-A (post-injection answer == the attacker's target).
Results + `summary.jsonl` land under `letta/results/{gpt4mini,llama}/`.
