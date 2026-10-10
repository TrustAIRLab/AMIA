# Defenses — setup

Four defense results. All reuse the existing release runners' outputs and the
shared defense templates in `../amia/defenses.py` + the judge client in
`../llm_judges/`.

```
defense/
├── aggregate_table_a.py          # A: appendix_defense_full (single-agent, 3 defenses)
├── benign_fp_consistency.py      # B: benign_fp_consistency
├── minja_detector_on_amia.py     # C: minja_defense_on_amia
├── aggregate_table_d.py          # D: appendix_defense_arch (consistency x memory archs)
├── prompts.py                    # MINJA input detector prompts (Targeted/General)
└── configs/{benign_fp,minja_detector}_configs.py
```

## A — appendix_defense_full (3 defenses × single-agent)
Already implemented by the release single-agent runners via their `DEFENSE` knob
(`none | paraphrase | relevance | consistency`), using the `amia/defenses.py`
templates. Run each of `single_agent/{qa,multiqa,react}_{llama,gpt4mini}.py`
once per `DEFENSE` value (`none` = Base), then:
```
python defense/aggregate_table_a.py
```
It reads `single_agent/results/summary.jsonl` and prints ASR-I/R/A + DEFpass per
(agent, attack, defense). (Paraphrase DEFpass is 1.0 by construction.)

## B — benign_fp_consistency
Needs the single-agent **Llama** `no_attack` 200-subset runs (QA/Multi-QA/ReAct).
```
python defense/benign_fp_consistency.py
```
Reads the benign stored responses and asks the Consistency judge (Llama + GPT)
whether each is consistent; FP = benign responses judged inconsistent / non-empty
(ReAct denominator ~42, since only `Finish[...]` responses are real writes).

## C — minja_defense_on_amia
MINJA's input-side detector on AMIA's injected queries. Needs the single-agent
runs (the runners now log each entry's `q_adv`); the cells use the best Semi-I /
No-I variant per agent + PI + benign.
```
python defense/minja_detector_on_amia.py
```
Reports flagged/N per (agent, family) under the Targeted and General detectors,
for the Llama and GPT-4.1-mini judges (benign = the detector's false-positive
rate).

## D — appendix_defense_arch (consistency × Mem0/A-MEM/MemoryOS)
Llama-only, N=200. Run each production-memory Llama runner twice — `DEFENSE=none`
and `DEFENSE=consistency` — for all three backends × QA/Multi-QA/ReAct (the
consistency gate drops memory writes judged inconsistent, using the same
`CONSISTENCY_TEMPLATE`), then:
```
python defense/aggregate_table_d.py
```
It reads `production_memory/{Mem0,AMem,MemoryOS}/results/summary.jsonl` and
reports baseline vs +Consistency ASR-A, Delta, and DEFpass per cell.

## Judges / backends
- GPT judge (B, C GPT column): `OPENAI_API_KEY` in a `.env` at the repo root.
- Llama judge (B, C Llama column) and all of D: a local vLLM server
  (`VLLM_BASE` / `LLAMA_MODEL_NAME`) and Llama-3.1-70B weights. Comment the Llama
  judge out of the B/C configs to run only the GPT judge.
