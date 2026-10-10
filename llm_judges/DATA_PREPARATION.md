# LLM-as-judge validations — setup

Two post-hoc validations that run an LLM judge over the **single-agent runs'
stored memory entries** (no new attacks). They read the per-entry
`*_intermediate.json` (and `*_final.json`) that `../single_agent/` already
writes, so you must run the relevant single-agent experiments first.

```
llm_judges/
├── prompts.py            # verbatim judge prompts (stealth targeted/general; semantic I/R)
├── judge_client.py       # shared OpenAI-compatible judge client (temp 0, 1-word verdict)
├── stealth/run_stealth.py    # table:stealth_detection  (Llama + GPT judges)
├── semantic/run_semantic.py  # table:semantic_validation (GPT-4.1-mini judge only)
└── configs/{stealth,semantic}_configs.py
```

## 1. Produce the single-agent corpus first
Both judges consume `../single_agent/results/<run>/<attack>/*_intermediate.json`.

- **Stealth** reads the **Llama** runs on the **200-target subset** (set
  `N_SAMPLE="200"` in the single-agent Llama configs), needing these attacks:
  `pi`, `discredit_restate` (QA Semi-I), `restate_elaboration` (Multi-QA Semi-I),
  `why_inquiry` (No-I), `fake_search_restate_elab` (ReAct Semi-I),
  `multiobs_verification` (ReAct No-I), and `no_attack` (benign). The default
  `ATTACK="all"` covers the QA/Multi-QA ones; make sure the ReAct run includes
  the two ReAct variants + `no_attack`. These land in
  `single_agent/results/{qa_200, multiqa_commonsenseqa_200, react_200}/<attack>/`.
- **Semantic** reads the **GPT-4.1-mini** runs (`single_agent/*_gpt4mini.py`) for
  the 9 cells (`pi`, the Semi-I/No-I variants above), in
  `single_agent/results/{qa, multiqa_commonsenseqa, react}/<attack>/`.

(The variant-dir → cell mapping is in the configs; adjust if your run tags
differ.)

## 2. Install / judges
```
pip install -r ../requirements.txt
```
- **GPT judge** (semantic, and the GPT column of stealth): `OPENAI_API_KEY` in a
  `.env` at the repo root.
- **Llama judge** (the Llama column of stealth): a local vLLM OpenAI-compatible
  server; set `VLLM_BASE` / `LLAMA_MODEL_NAME`. Comment the Llama judge out of
  `configs/stealth_configs.py::JUDGES` to run only the GPT judge.

## 3. Run
From the repo root:
```
python llm_judges/stealth/run_stealth.py     # detection rate per (agent, category), both judges
python llm_judges/semantic/run_semantic.py    # ROUGE vs Semantic ASR-I/ASR-R, N=1800 totals
```
Set `LIMIT > 0` in a config for a quick smoke (first LIMIT entries/cell; outputs
isolated under `results_smoke/`).

## Notes
- The stealth corpus is built from the **Llama** entries and scored by **both**
  judges (the GPT judge audits Llama-stored entries), matching the paper.
- ReAct benign FP uses only entries that reached a `Finish[...]` action
  (~42/200); the rest never write a real memory entry. The `Finish[...]` wrapper
  is stripped so the judge sees the same surface on attack and benign entries.
- Semantic validates attack families only (no benign); ASR-A is read from the
  single-agent `*_final.json`, not produced by the judge.
