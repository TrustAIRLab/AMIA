"""
Table C (table:minja_defense_on_amia): MINJA's prompt-level detector vs AMIA.

MINJA's defense inspects the attacker's input query `q_adv` (logged by the
single-agent runners as the `q_adv` field) and flags injection attempts. For
each (agent, family) we report how many of N queries each detector (Targeted,
General) flags; on benign queries this is the detector's false-positive rate.
Run under both judges (Llama + GPT-4.1-mini).

All knobs in `configs/minja_detector_configs.py`. Run from the repo root (after
the single-agent runs that log `q_adv` exist):

    python defense/minja_detector_on_amia.py
"""
import os
import sys
import glob
import json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import minja_detector_configs as C
from prompts import MINJA_TARGETED_TEMPLATE, MINJA_GENERAL_TEMPLATE
from llm_judges.judge_client import make_client, parse_bool, judge_map

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

TEMPLATES = {"targeted": MINJA_TARGETED_TEMPLATE, "general": MINJA_GENERAL_TEMPLATE}
AGENTS = ["QA", "Multi-QA", "ReAct"]
FAMILIES = ["PI", "Semi-I", "No-I", "benign"]


def load_qadv(variant_dir):
    matches = glob.glob(os.path.join(C.SINGLE_AGENT_RESULTS, variant_dir, "*_intermediate.json"))
    if not matches:
        return None
    data = json.load(open(matches[0]))
    if C.LIMIT > 0:
        data = data[:C.LIMIT]
    out = [str(r.get("q_adv", "")) for r in data if r.get("q_adv")]
    return out


def main():
    results_root = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR
    os.makedirs(results_root, exist_ok=True)
    # Preload corpora once.
    corpora = {}
    for agent, family, variant_dir in C.CELLS:
        q = load_qadv(variant_dir)
        if q is None:
            print(f"[warn] missing q_adv for {agent}/{family}: {variant_dir}")
            continue
        corpora[(agent, family)] = q

    summary = {"judges": {}}
    for jcfg in C.JUDGES:
        client = make_client(base_url=jcfg["base_url"], api_key=jcfg["api_key"])
        cells = {}
        print(f"\n=== MINJA detector on AMIA ({jcfg['name']} judge) — flagged/N ===")
        header = f"{'agent':<9} {'family':<8}" + "".join(f"{d:>14}" for d in C.DETECTORS)
        print(header)
        for agent in AGENTS:
            for fam in FAMILIES:
                q = corpora.get((agent, fam))
                if q is None:
                    continue
                row = {}
                for det in C.DETECTORS:
                    tpl = TEMPLATES[det]
                    verdicts = [parse_bool(t) for t in
                                judge_map(client, jcfg["model"], [tpl.format(q_adv=x) for x in q], C.MAX_WORKERS)]
                    decided = [v for v in verdicts if v is not None]
                    flagged = sum(1 for v in decided if v is True)
                    n = len(decided)
                    row[det] = {"flagged": flagged, "n": n,
                                "rate": round(100.0 * flagged / n, 1) if n else None}
                cells[f"{agent}/{fam}"] = row
                print(f"{agent:<9} {fam:<8}" +
                      "".join(f"{row[d]['flagged']:>5}/{row[d]['n']:<3}({row[d]['rate']:>5})" for d in C.DETECTORS))
        summary["judges"][jcfg["name"]] = cells
    with open(os.path.join(results_root, "minja_detector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
