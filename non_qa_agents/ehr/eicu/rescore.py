"""Recompute eICU ASR-A offline from already-saved agent log dirs, without
re-running the LLM. Useful after changing the eval logic. Reads the per-pair
`logs/4/` dirs under the results tree and rewrites each `result.json` + the
aggregated `summary.json`.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from configs import ehr_eicu_configs as C
from ehr_eicu_attack import eval_test_only, MINJA_PAPER_PAIRS

# Canonical victim -> target map (authoritative; deriving target from the
# directory name is lossy for multi-word drug names).
VICTIM_TO_TARGET = {v: t for v, t in MINJA_PAPER_PAIRS}
OUT_BASE = Path((C.RESULTS_DIR + "_smoke") if C.LIMIT > 0 else C.RESULTS_DIR)


def main():
    by_variant = defaultdict(list)
    for variant_dir in sorted(OUT_BASE.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        for pair_dir in sorted(variant_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            ds = json.load(open(pair_dir / "dataset.json"))
            victim = next((e["value"]["drug_name"] for e in ds if e["id"].startswith("test_")), None)
            target = VICTIM_TO_TARGET.get(victim, pair_dir.name.split("__", 1)[-1].replace("_", " "))
            if (pair_dir / "result.json").exists():
                old = json.load(open(pair_dir / "result.json"))
                victim = old["victim"]
                target = VICTIM_TO_TARGET.get(victim, old["target"])
            n_succ, n_total, per_q = eval_test_only(pair_dir / "logs" / "4", victim, target)
            if n_total == 0:
                continue
            res = {"variant": variant_dir.name, "victim": victim, "target": target,
                   "test_succ": n_succ, "test_total": n_total,
                   "asr_a": n_succ / n_total, "per_query": per_q}
            with open(pair_dir / "result.json", "w") as f:
                json.dump(res, f, indent=2)
            by_variant[variant_dir.name].append(res)
            print(f"  [{variant_dir.name:<12}] {victim[:22]:<22} -> {target[:22]:<22}  "
                  f"ASR-A={n_succ}/{n_total}={n_succ/n_total*100:.1f}%")

    summary = {"variants": {}}
    for v, rows in by_variant.items():
        ts = sum(r["test_succ"] for r in rows)
        tt = sum(r["test_total"] for r in rows)
        summary["variants"][v] = {"aggregated_asr_a": round(ts / tt, 4) if tt else 0,
                                  "test_succ": ts, "test_total": tt, "n_pairs": len(rows),
                                  "per_pair_asr_a": [r["asr_a"] for r in rows]}
    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Re-scored ASR-A ===")
    for v, r in sorted(summary["variants"].items(), key=lambda kv: kv[1]["aggregated_asr_a"], reverse=True):
        print(f"  {v:<12} agg={r['aggregated_asr_a']*100:5.2f}% ({r['test_succ']}/{r['test_total']})")


if __name__ == "__main__":
    main()
