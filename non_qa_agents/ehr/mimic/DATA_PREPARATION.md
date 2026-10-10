# EHRAgent / MIMIC-III — data & framework setup

The clinical agent is a vendored MIMIC-aware EHRAgent app (`ehragent/`), driven
as a subprocess by `ehr_mimic_attack.py`. We do not ship the MIMIC-III data (it
is a credentialed PhysioNet dataset); obtain it and place the files where the
app and driver expect.

## 1. Install the framework
```
pip install pyautogen sentence-transformers pandas openai python-dotenv
```
The EHRAgent app uses AutoGen (`autogen`) for the code-generation agent and a
SentenceTransformer embedder for its long-term memory. Docker is disabled
(`AUTOGEN_USE_DOCKER=False`, set automatically by the driver).

## 2. Get the MIMIC-III data (PhysioNet, credentialed)
MIMIC-III requires a credentialed PhysioNet account and completion of the data
use agreement (https://physionet.org/content/mimiciii/). From the EHRSQL /
EHRAgent MIMIC-III setup, place three files under:

```
non_qa_agents/ehr/mimic/ehrsql-ehragent/mimic_iii/
├── mimic_iii.db        # the SQLite database the agent queries
├── patient_only.json   # query templates (Patient ID type) used to build queries
└── PATIENTS.csv        # the patient-ID pool (SUBJECT_ID column)
```
The driver reads `patient_only.json` (templates) and `PATIENTS.csv` (to pick
victim/target patient pairs); the app opens `mimic_iii.db` from this directory
(via a path relative to `ehragent/`). The config paths are set in
`configs/ehr_mimic_configs.py`.

## 3. Backbone
Set in `configs/ehr_mimic_configs.py`:
- `BACKEND="openai"`, `MODEL="gpt-4.1-mini"` — GPT via `OPENAI_API_KEY` in a
  `.env` at the repo root (the vendored app routes `gpt-*` names to OpenAI).
- `BACKEND="vllm"`, `MODEL=<served llama name>`, `VLLM_BASE=<url>` — local Llama
  via a vLLM OpenAI-compatible server.

## 4. Run
From the repo root:
```
python non_qa_agents/ehr/mimic/ehr_mimic_attack.py
```
Defaults: 9 patient pairs x 30 test queries, 4 attack families (MINJA PSS, PI,
Semi-I, No-I). Set `LIMIT > 0` in the config for a quick smoke (3 pairs, few
tests; outputs under `results_smoke/`). `score_results.py` re-derives
`summary.json` offline from the per-pair `result.json` files if needed.
```
```
