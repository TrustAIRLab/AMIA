# RAP / WebShop — data & server setup

The RAP shopping agent runs against a **WebShop** environment. We do not ship the
WebShop product catalog; download it once and serve it locally with the bundled
`mock_webshop.py` (a lightweight Flask + BM25 stand-in for the full WebShop
server — no Node/React stack needed).

## 1. Get the WebShop product data
Obtain the WebShop item files from the official WebShop release
(princeton-nlp/WebShop, `data/` directory):

- `items_shuffle.json`      (full catalog; contains the paper's 9 target products)
- `items_ins_v2.json`
- `items_human_ins.json`

(The 1000-product subsets `items_shuffle_1000.json` / `items_ins_v2_1000.json`
also work for a quick check but will not contain every target product.)

Place them under:

```
non_qa_agents/rap/webshop_setup/WebShop/data/
    items_shuffle.json
    items_ins_v2.json
    items_human_ins.json
```

Or point the server elsewhere with the `WEBSHOP_DATA_DIR` env var (it should
contain a `data/` subdirectory with the files above).

## 2. Start the mock WebShop server (serves on :3000)
```
python non_qa_agents/rap/mock_webshop.py            # uses WEBSHOP_PORT (default 3000)
```
Leave it running in a separate terminal. The attack runner talks to it over
`WEBSHOP_URL` (default `http://localhost:3000`).

## 3. Run the attack
From the repo root, with `OPENAI_API_KEY` in a `.env` at the repo root:
```
python non_qa_agents/rap/rap_gpt.py
```
Defaults (9 victim/target pairs x 30 test queries, 4 attack families) reproduce
the paper's web-shopping cell. Set `LIMIT > 0` in `configs/rap_gpt_configs.py`
for a quick end-to-end smoke test (one pair, few injects/tests; outputs under
`results_smoke/`).

## Vendored assets (shipped, no download needed)
`rap_assets/` contains the RAP WebShop few-shot prompt, the 9 victim/target
pairs, and the WebShop instruction pool — these are small and included.
