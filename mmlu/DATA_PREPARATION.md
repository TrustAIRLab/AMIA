# MMLU head-to-head — data setup

AMIA vs MINJA on MMLU (N=200). The MMLU agent uses the same flat key-value DPR
memory as the Multi-QA agent, but with a **SciQ** base corpus as the benign
memory the attacker's injections compete against.

## 1. MMLU questions (already shipped)
`mmlu/data/mmlu_n200.json` is included — the deterministic 200-question MMLU
subset (across 54 subjects) with a pre-selected wrong-answer target per question
(`target_letter` / `target_text`). No download needed.

## 2. SciQ base corpus (already shipped)
The memory's benign base corpus is SciQ (public: Hugging Face `allenai/sciq`).
Both the corpus and its prebuilt DPR embedding cache are shipped under `../data/`:
- `data/sciq/train-00000-of-00001.parquet` (the `support` passages)
- `data/multiqa_embeddings/embeddings_sciq_dpr-ctx_encoder-single-nq-base.pkl`

so no download or embedding build is needed (delete the cache to force a
recompute). The corpus path / name are set in `configs/mmlu_*_configs.py`
(`CORPUS_FILE`, `DATABASE_NAME`). The embedder `dpr-ctx_encoder-single-nq-base`
downloads on first use.

## 3. Install / backbone
```
pip install -r ../requirements.txt
```
- GPT: `OPENAI_API_KEY` in a `.env` at the repo root (`mmlu_gpt4mini.py`).
- Llama: local vLLM weights at `MODEL_PATH` (`mmlu_llama.py`).

## 4. Run
From the repo root (one invocation runs PI, Semi-I, No-I, and MINJA):
```
python mmlu/mmlu_gpt4mini.py
python mmlu/mmlu_llama.py
```
PI/Semi-I/No-I are 1x cost; MINJA runs 5 progressively-shortened rounds and the
best-of-5 ASR-A is reported. Set `LIMIT > 0` in a config for a quick smoke test
(outputs isolated under `results_smoke/`).
