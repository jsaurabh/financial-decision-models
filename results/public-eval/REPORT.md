# Public decision evaluation

Status: complete. Frozen corpus: 26,196 decisions. No training, calibration fitting, input truncation or candidate reduction.

Gemma and Qwen use strict math SDPA with frozen checkpoints and temperatures. The Gemma inference envelope is extended to eleven candidates and 8,192 tokens; this exceeds the original release validation envelope. Each model gets its original prompt and actual tokenizer. Unsupported or failed cases remain in the denominator below.

| Suite | Cases | Causal Gemma | Joint Gemma | Qwen 27B | Jev |
|---|---:|---:|---:|---:|---:|
| contractnli | 2091 | 51.17% (100.0%) | 47.68% (100.0%) | 78.96% (100.0%) | 75.37% (100.0%) |
| ragtruth | 2700 | 64.26% (100.0%) | 64.44% (100.0%) | 87.30% (100.0%) | 78.26% (100.0%) |
| rjudge | 571 | 42.73% (100.0%) | 45.71% (100.0%) | 85.81% (100.0%) | 87.22% (100.0%) |
| routerbench_0shot_cost_aware | 2500 | 16.84% (100.0%) | 18.20% (100.0%) | 16.20% (100.0%) | 16.56% (100.0%) |
| routerbench_0shot_quality | 2500 | 68.44% (100.0%) | 67.36% (100.0%) | 81.56% (100.0%) | 80.36% (100.0%) |
| routerbench_5shot_cost_aware | 2500 | 10.16% (100.0%) | 8.88% (100.0%) | 8.36% (100.0%) | 13.20% (100.0%) |
| routerbench_5shot_quality | 2500 | 77.20% (100.0%) | 67.28% (100.0%) | 84.44% (100.0%) | 83.44% (100.0%) |
| tsbench_agentdojo-traj | 1220 | 66.89% (89.8%) | 66.89% (89.8%) | 87.95% (89.8%) | 93.36% (100.0%) |
| tsbench_agentharm-traj | 731 | 28.73% (100.0%) | 28.04% (100.0%) | 42.00% (100.0%) | 43.64% (100.0%) |
| tsbench_asb-traj | 5231 | 51.23% (100.0%) | 51.58% (100.0%) | 53.64% (100.0%) | 56.60% (100.0%) |
| when2call | 3652 | 38.31% (100.0%) | 39.13% (100.0%) | 81.11% (100.0%) | 81.82% (100.0%) |

Entries show accuracy with missing predictions counted wrong, followed by coverage in parentheses. RouterBench entries are tie-aware oracle-optimal rates; realized quality, cost, utility and regret are the primary routing results in the detailed JSON. Different tasks should not be collapsed into one general-quality rank.

## Identical common-valid inputs

| Suite | Common cases | Causal | Joint | Qwen | Jev |
|---|---:|---:|---:|---:|---:|
| contractnli | 2091 | 51.17% | 47.68% | 78.96% | 75.37% |
| ragtruth | 2700 | 64.26% | 64.44% | 87.30% | 78.26% |
| rjudge | 571 | 42.73% | 45.71% | 85.81% | 87.22% |
| routerbench_0shot_cost_aware | 2500 | 16.84% | 18.20% | 16.20% | 16.56% |
| routerbench_0shot_quality | 2500 | 68.44% | 67.36% | 81.56% | 80.36% |
| routerbench_5shot_cost_aware | 2500 | 10.16% | 8.88% | 8.36% | 13.20% |
| routerbench_5shot_quality | 2500 | 77.20% | 67.28% | 84.44% | 83.44% |
| tsbench_agentdojo-traj | 1096 | 74.45% | 74.45% | 97.90% | 94.62% |
| tsbench_agentharm-traj | 731 | 28.73% | 28.04% | 42.00% | 43.64% |
| tsbench_asb-traj | 5231 | 51.23% | 51.58% | 53.64% | 56.60% |
| when2call | 3652 | 38.31% | 39.13% | 81.11% | 81.82% |

Resolved Jev providers: `{"jev-1.14.0": 26196}`.

See [full metrics](summary.json) for valid-only classification F1, safety misses/false blocks under both severity conventions, binary positive-class scores, contract document-exact match, routing outcomes and exploratory paired Gemma intervals. API failures, over-context inputs and OOM failures are explicitly recorded.

## Reproduction and limits

- [Frozen protocol](../../PUBLIC_EVAL_PROTOCOL.md) and [data manifest](../../data/public-eval-manifest.json).
- [Pinned private output artifacts](artifacts.json).
- [Private predictions](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval).

RouterBench is a hash-selected initial subset, preserving both shot regimes/objectives and all eleven candidates. Other named source test records are retained. R-Judge contains 571 records at the pinned revision although its paper states 569. ContractNLI has seventeen decisions per contract. TS-Bench and R-Judge here evaluate static judgments, not end-to-end agent behavior.

The published public sources may have appeared in pretraining or earlier upstream fine-tuning; Qwen has previously been evaluated on Decision Index. This is not a new contamination-free or private production benchmark. Raw source contexts are not republished. Source-family overlap limits the apparent number of independent examples.

A collector bug initially overwrote local Jev output with older remote checkpoints. The collector now uses a separate private cache. After validating saved input hashes, candidate order and targets, 23,239 records were retained and only 2,957 missing records were re-requested. No cases were selected for retry based on prediction correctness.
