# MINJA baseline (QA head-to-head) — setup

This folder runs the **MINJA** iterative attack (the prior query-only baseline we
compare against) on the same three single-agent QA agents as `../single_agent/`:
StrategyQA (QA), CommonsenseQA (Multi-QA), and the ReAct agent. It produces the
MINJA column of the QA head-to-head table; the AMIA column is produced by the
`../single_agent/` runners.

## 1. Data (already shipped)
These runners reuse the shared corpora and indices under `../data/` — the same
files the single-agent runners use, so no extra download is needed:
- `strategyqa_train.json`, `strategyqa_train_paragraphs.json`, `contradiction_results.json`
- `commonsenseqa/` (`multiple_choice_questions.pkl`, `correct_answers.pkl`,
  `misleading_answers.pkl`, `train.parquet`) — the Multi-QA **question set**
- `sciq/train-00000-of-00001.parquet` — the Multi-QA MINJA **base memory corpus**
  (see the note below)
- `react_prompts.json`, `sampled_indices.json`
- the DPR embedding caches under `qa_embeddings/` and `multiqa_embeddings/`
  (including `embeddings_sciq_...pkl`)

Note on the Multi-QA base corpus: the paper's MINJA Multi-QA run keys its benign
base memory on **SciQ** (the MINJA baseline's default), while the AMIA Multi-QA
run (`../single_agent/`) uses the **CommonsenseQA** base. The two head-to-head
sides therefore use different base corpora by design; these configs reproduce
that exactly (`DATABASE_NAME="sciq"` for the MINJA Multi-QA base).

## 2. Install
```
pip install -r ../requirements.txt
```
The DPR flat-KV memory uses `transformers` + `torch` (the embedder
`dpr-ctx_encoder-single-nq-base` downloads on first use). The `*_llama.py`
runners additionally need `vllm` and local Llama-3.1-70B weights
(`MODEL_PATH` in the config); the `*_gpt4mini.py` runners need `OPENAI_API_KEY`
in a `.env` at the repo root.

## 3. Run
From the repo root (each runs 5 MINJA rounds and reports best-of-5 ASR-A):
```
python minja_qa/qa_gpt4mini.py          # QA, GPT-4.1-mini
python minja_qa/multiqa_gpt4mini.py     # Multi-QA, GPT-4.1-mini
python minja_qa/react_gpt4mini.py       # ReAct, GPT-4.1-mini
# …and the *_llama.py counterparts for Llama-3.1-70B
```
Set `ABLATION = "input_key"` in a config to produce the input-key memory row
(keys memory on the attacker query instead of the response); the default
`"none"` is the answer-based row. Set `LIMIT > 0` for a quick smoke test
(outputs isolated under `results_smoke/`).

Sample sizes (paper): Llama uses the full set (QA/ReAct N=2270, Multi-QA N=986);
GPT uses the shared 200-target subset (Multi-QA's effective count is ~195, since
a few sampled indices fall outside the local corpus).
