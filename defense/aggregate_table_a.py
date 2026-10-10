"""
Table A (table:appendix_defense_full): aggregate the single-agent defense runs.

No new experiment -- the release single_agent runners already implement the three
defenses via their `DEFENSE` config knob. Run each of
  single_agent/{qa,multiqa,react}_{llama,gpt4mini}.py
with DEFENSE in {none, paraphrase, relevance, consistency} (none = Base), then
this script reads the rows they append to `single_agent/results/summary.jsonl`
and prints ASR-I / ASR-R / ASR-A + DEFpass per (agent, attack) across the four
conditions. Run from the repo root:

    python defense/aggregate_table_a.py
"""
import os
import sys
import json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # new_published/
SUMMARY = os.path.join(_ROOT, "single_agent", "results", "summary.jsonl")
CONDITIONS = ["none", "paraphrase", "relevance", "consistency"]


def main():
    if not os.path.isfile(SUMMARY):
        print(f"No single_agent summary at {SUMMARY}. Run the single_agent runners "
              f"with DEFENSE in {CONDITIONS} first (see DATA_PREPARATION.md).")
        return
    rows = {}
    for line in open(SUMMARY):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ablation", "none") != "none":
            continue   # ablation runs are reported separately
        key = (r.get("model", "?"), r.get("subset", "?"), r["agent"], r.get("attack"),
               r.get("defense", "none"))
        rows[key] = r

    by_key = defaultdict(dict)
    for (model, subset, agent, attack, defense), r in rows.items():
        by_key[(model, subset, agent, attack)][defense] = r

    print(f"{'model':<14} {'subset':<6} {'agent':<9} {'attack':<20} {'defense':<12} "
          f"{'ASR-I':>6} {'ASR-R':>6} {'ASR-A':>6} {'DEFpass':>8}")
    for (model, subset, agent, attack) in sorted(by_key):
        for cond in CONDITIONS:
            r = by_key[(model, subset, agent, attack)].get(cond)
            if not r:
                continue
            # Paraphrase rewrites (never gates), so DEFpass is 1.0 by construction.
            defpass = 1.0 if cond == "paraphrase" else r.get("defense_pass_rate")
            dp = "--" if (cond == "none" or defpass is None) else f"{defpass*100:.1f}"
            print(f"{model:<14} {subset:<6} {agent:<9} {attack:<20} {cond:<12} "
                  f"{r['asr_i']*100:6.1f} {r['asr_r']*100:6.1f} {r['asr_a']*100:6.1f} {dp:>8}")


if __name__ == "__main__":
    main()
