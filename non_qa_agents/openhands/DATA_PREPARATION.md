# OpenHands code-summary — framework & data setup

This experiment drives a real **OpenHands** agent and a **Mem0** memory over the
**HumanEval** dataset. None of these are shipped; install the frameworks and let
`datasets` fetch HumanEval on first run.

## 1. Install the frameworks
```
pip install openhands-ai mem0ai datasets rouge-score sentence-transformers qdrant-client
```
(`openhands-ai` provides the `openhands.sdk` used for the agent; `mem0ai` is the
memory store; `sentence-transformers` provides the local all-MiniLM-L6-v2
embedder Mem0 uses.)

## 2. HumanEval
No manual download: the runner calls
`datasets.load_dataset("openai/openai_humaneval", split="test")`, which fetches
and caches the 164 problems automatically (needs network on first run).

## 3. Backbone
- **GPT-4.1-mini (default):** put `OPENAI_API_KEY` in a `.env` at the repo root.
  Uses `configs/openhands_gpt4mini_configs.py`.
- **Llama-3.1-70B:** start a vLLM OpenAI-compatible server and point
  `VLLM_BASE_URL` at it, then run with `configs/openhands_llama_configs.py`
  (import it in place of the GPT config at the top of
  `openhands_code_summary.py`).

## 4. Run
From the repo root:
```
python non_qa_agents/openhands/openhands_code_summary.py
```
Defaults run all attack families (PI x5 framings, Semi-I, No-I x3, MINJA, plus
the no-attack baseline) over the full 164 HumanEval functions. Set `LIMIT > 0` in
the config for a quick smoke test (first LIMIT functions; outputs under
`results_smoke/`).
