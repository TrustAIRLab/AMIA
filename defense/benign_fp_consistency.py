"""
Table B (table:benign_fp_consistency): false-positive rate of the Consistency
defense on benign writes.

Reads the single-agent `no_attack` stored responses, asks the Consistency judge
whether each is consistent with the model's own knowledge, and reports the
false-positive rate per agent = (benign responses judged inconsistent) / (non-
empty responses). ReAct counts only entries that reached a Finish action.

All knobs in `configs/benign_fp_configs.py`; GPT judge reads OPENAI_API_KEY from
a `.env` at the repo root, Llama judge a vLLM server. Run from the repo root
(after the single-agent `no_attack` runs exist):

    python defense/benign_fp_consistency.py
"""
import os
import sys
import glob
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from configs import benign_fp_configs as C
from amia.defenses import CONSISTENCY_TEMPLATE
from llm_judges.judge_client import make_client, parse_bool, judge_map

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))


def load_benign(variant_dir, qfield, rfield):
    matches = glob.glob(os.path.join(C.SINGLE_AGENT_RESULTS, variant_dir, "*_intermediate.json"))
    if not matches:
        return None
    data = json.load(open(matches[0]))
    if C.LIMIT > 0:
        data = data[:C.LIMIT]
    out = []
    for rec in data:
        resp = rec.get(rfield, "")
        if not resp or not isinstance(resp, str) or not resp.strip():
            continue   # ReAct: no Finish -> no memory write -> skip
        if rfield == "final_action" and not resp.strip().startswith("Finish["):
            continue
        out.append((str(rec.get(qfield, "")), resp))
    return out


def main():
    results_root = (C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR
    os.makedirs(results_root, exist_ok=True)
    summary = {"judges": {}}
    for jcfg in C.JUDGES:
        client = make_client(base_url=jcfg["base_url"], api_key=jcfg["api_key"])
        print(f"\n=== Consistency benign-FP ({jcfg['name']} judge) ===")
        per_agent, fp_sum, n_sum = {}, 0, 0
        for agent, variant_dir, qfield, rfield in C.CELLS:
            entries = load_benign(variant_dir, qfield, rfield)
            if entries is None:
                print(f"[warn] missing {agent}: {variant_dir}"); continue
            prompts = [CONSISTENCY_TEMPLATE.format(q, a) for q, a in entries]
            verdicts = [parse_bool(t) for t in judge_map(client, jcfg["model"], prompts, C.MAX_WORKERS)]
            decided = [v for v in verdicts if v is not None]
            # Consistency judge: True = consistent (keep); False = inconsistent -> FP.
            fp = sum(1 for v in decided if v is False)
            n = len(decided)
            per_agent[agent] = {"fp": fp, "n": n, "fp_rate": round(100.0 * fp / n, 1) if n else None}
            fp_sum += fp; n_sum += n
            print(f"  {agent:<9} FP = {fp}/{n} = {per_agent[agent]['fp_rate']}%")
        mean = round(100.0 * fp_sum / n_sum, 1) if n_sum else None
        print(f"  {'Mean':<9} {mean}%")
        summary["judges"][jcfg["name"]] = {"per_agent": per_agent, "mean_fp_rate": mean}
    with open(os.path.join(results_root, "benign_fp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
