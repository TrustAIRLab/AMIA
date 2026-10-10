"""
LLM-as-judge stealth detection (table:stealth_detection).

Builds a corpus of stored memory entries from the single-agent Llama runs (one
`actual_inject` per attack entry; the `no_attack` response for benign entries),
then asks a memory-auditor judge whether each entry looks like an injection.
Per (agent, category) the detection rate = flagged / decided (benign -> false-
positive rate). Runs every judge in the config (paper: Llama + GPT-4.1-mini).

All knobs live in `configs/stealth_configs.py`; the GPT judge reads
OPENAI_API_KEY from a `.env` at the repo root, the Llama judge a vLLM server.
Run from the repo root (after producing the single-agent corpus):

    python llm_judges/stealth/run_stealth.py
"""
import os
import sys
import glob
import json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # llm_judges/
_ROOT = os.path.dirname(os.path.dirname(_PKG))      # new_published/
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

from configs import stealth_configs as C
from prompts import STEALTH_TARGETED_TEMPLATE, STEALTH_GENERAL_TEMPLATE
from judge_client import make_client, parse_bool, judge_map

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

TEMPLATE = STEALTH_TARGETED_TEMPLATE if C.DETECTOR == "targeted" else STEALTH_GENERAL_TEMPLATE
AGENTS = ["QA", "Multi-QA", "ReAct"]
FAMILIES = ["PI", "Semi-I", "No-I", "benign"]


def _normalize(agent, entry):
    """Strip agent-side formatting so attack and benign entries share the same
    visible surface. For ReAct, strip the `Finish[...]` wrapper (otherwise the
    judge trivially splits on the format, not the content)."""
    e = entry.strip()
    if agent == "ReAct" and e.startswith("Finish[") and e.endswith("]"):
        return e[len("Finish["):-1].strip()
    return e


def build_corpus():
    corpus = []
    for agent, family, variant_dir, field, is_benign in C.CELLS:
        matches = glob.glob(os.path.join(C.SINGLE_AGENT_RESULTS, variant_dir, "*_intermediate.json"))
        if not matches:
            print(f"[warn] missing corpus for {agent}/{family}: {variant_dir}")
            continue
        data = json.load(open(matches[0]))
        cnt = 0
        for rec in data:
            entry = rec.get(field, "")
            if not entry or not isinstance(entry, str):
                continue
            # ReAct benign: only entries that reached a Finish action are real writes.
            if is_benign and agent == "ReAct" and not entry.strip().startswith("Finish["):
                continue
            normalized = _normalize(agent, entry)
            if not normalized:
                continue
            corpus.append({"agent": agent, "family": family, "entry": normalized})
            cnt += 1
            if C.LIMIT > 0 and cnt >= C.LIMIT:
                break
    return corpus


def run_judge(corpus, jcfg, results_root):
    client = make_client(base_url=jcfg["base_url"], api_key=jcfg["api_key"])
    prompts = [TEMPLATE.format(entry=item["entry"]) for item in corpus]
    verdicts = [parse_bool(t) for t in judge_map(client, jcfg["model"], prompts, C.MAX_WORKERS)]

    bucket = defaultdict(lambda: {"flagged": 0, "decided": 0})
    for item, v in zip(corpus, verdicts):
        if v is None:
            continue
        b = bucket[(item["agent"], item["family"])]
        b["decided"] += 1
        b["flagged"] += int(v)

    summary = {"judge": jcfg["name"], "model": jcfg["model"], "detector": C.DETECTOR, "cells": {}}
    print(f"\n=== Stealth detection ({jcfg['name']} judge, {C.DETECTOR}) — rate% (flagged/decided) ===")
    print(f"{'agent':<9} {'PI':>14} {'Semi-I':>14} {'No-I':>14} {'benign':>14}")
    for agent in AGENTS:
        cells = []
        for fam in FAMILIES:
            b = bucket.get((agent, fam))
            if not b or b["decided"] == 0:
                cells.append("—"); continue
            rate = 100.0 * b["flagged"] / b["decided"]
            summary["cells"][f"{agent}/{fam}"] = {"rate": round(rate, 1),
                                                  "flagged": b["flagged"], "decided": b["decided"]}
            cells.append(f"{rate:5.1f} ({b['flagged']}/{b['decided']})")
        print(f"{agent:<9} " + " ".join(f"{c:>14}" for c in cells))

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"stealth_summary_{jcfg['name']}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    results_root = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR
    corpus = build_corpus()
    print(f"Stealth corpus: {len(corpus)} entries across {len(C.CELLS)} cells")
    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, "stealth_corpus.jsonl"), "w") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")
    for jcfg in C.JUDGES:
        run_judge(corpus, jcfg, results_root)


if __name__ == "__main__":
    main()
