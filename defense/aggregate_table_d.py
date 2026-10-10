"""
Table D (table:appendix_defense_arch): the Consistency defense across the three
production-memory architectures.

No new experiment -- run the production_memory Llama runners twice (DEFENSE=none
and DEFENSE=consistency) for Mem0 / A-MEM / MemoryOS on QA/Multi-QA/ReAct, then
this script reads each backend's `results/summary.jsonl` and reports, per
(agent, backend, attack): baseline ASR-A, +Consistency ASR-A, Delta, and DEFpass.
Run from the repo root:

    python defense/aggregate_table_d.py
"""
import os
import sys
import json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # new_published/
PM = os.path.join(_ROOT, "production_memory")
SUMMARIES = [os.path.join(PM, b, "results", "summary.jsonl") for b in ("Mem0", "AMem", "MemoryOS")]


def main():
    # rows[(model, backend, agent, attack)][defense] = row
    rows = defaultdict(dict)
    found = False
    for sp in SUMMARIES:
        if not os.path.isfile(sp):
            continue
        found = True
        for line in open(sp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            backend = r.get("memory_backend", "?")
            key = (r.get("model", "?"), backend, r.get("agent"), r.get("attack"))
            rows[key][r.get("defense", "none")] = r
    if not found:
        print("No production_memory summaries found. Run the Mem0/AMem/MemoryOS Llama "
              "runners with DEFENSE=none and DEFENSE=consistency first (see DATA_PREPARATION.md).")
        return

    print(f"{'model':<14} {'backend':<9} {'agent':<9} {'attack':<22} {'base':>6} {'+cons':>6} {'delta':>7} {'DEFpass':>8}")
    deltas = []
    for (model, backend, agent, attack) in sorted(rows):
        base = rows[(model, backend, agent, attack)].get("none")
        cons = rows[(model, backend, agent, attack)].get("consistency")
        if not base or not cons:
            continue
        b, c = base["asr_a"] * 100, cons["asr_a"] * 100
        delta = c - b
        deltas.append(delta)
        dp = cons.get("defense_pass_rate")
        dp_s = f"{dp*100:.1f}" if dp is not None else "--"
        print(f"{model:<14} {backend:<9} {agent:<9} {attack:<22} {b:6.1f} {c:6.1f} {delta:+7.1f} {dp_s:>8}")
    if deltas:
        print(f"\nMean Delta ASR-A = {sum(deltas)/len(deltas):+.1f}%  over {len(deltas)} cells")


if __name__ == "__main__":
    main()
