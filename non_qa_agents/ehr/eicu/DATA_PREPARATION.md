# EHRAgent / eICU — data & framework setup

The clinical agent is a vendored eICU-aware EHRAgent app (`ehragent_eicu/`),
driven as a subprocess by `ehr_eicu_attack.py`. We do not ship the eICU database
(it is a credentialed PhysioNet dataset); obtain it and place it where the app
expects.

## 1. Install the framework
```
pip install pyautogen sentence-transformers openai python-dotenv
```
The EHRAgent app uses AutoGen (`autogen`) for the code-generation agent and a
SentenceTransformer embedder for its long-term memory. Docker is disabled
(`AUTOGEN_USE_DOCKER=False`, set automatically by the driver).

## 2. Get the eICU database (PhysioNet, credentialed)
eICU-CRD requires a credentialed PhysioNet account and completion of the data
use agreement (https://physionet.org/content/eicu-crd/). Build the SQLite
database used by the EHRSQL/EHRAgent eICU setup and place it at:

```
non_qa_agents/ehr/eicu/ehragent_eicu/eicu_real.db
```
(The app opens this path relative to `ehragent_eicu/`.)

The eICU **query templates** (`drug_only.json`) are small and already vendored
next to the driver — no download needed for those.

## 3. Backbone
Set in `configs/ehr_eicu_configs.py`:
- `BACKEND="openai"`, `MODEL="gpt-4.1-mini"` — GPT via `OPENAI_API_KEY` in a
  `.env` at the repo root (the driver sets `OPENAI_BACKEND=1` for the subprocess).
- `BACKEND="vllm"`, `MODEL=<served llama name>`, `VLLM_BASE=<url>` — local Llama
  via a vLLM OpenAI-compatible server.

## 4. Run
From the repo root:
```
python non_qa_agents/ehr/eicu/ehr_eicu_attack.py
```
Defaults: 9 drug pairs x 30 test queries, 4 attack families (MINJA PSS, PI,
Semi-I, No-I). Set `LIMIT > 0` in the config for a quick smoke (3 pairs, few
tests; outputs under `results_smoke/`). `rescore.py` re-derives `summary.json`
offline from the per-pair `result.json` files if needed.
```
```
