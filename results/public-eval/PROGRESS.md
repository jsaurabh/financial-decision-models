# Interim public evaluation

Causal Gemma and Jev each have all 26,196 expected unique case IDs. Canonical input hashes, candidate order and reference labels were checked against the sealed corpus. Joint Gemma is still running; Qwen is partial after runtime/connectivity interruptions. This is not a final four-model comparison.

| Task | Cases | Causal Gemma | Jev 1.14.0 | Gemma coverage |
|---|---:|---:|---:|---:|
| contractnli | 2091 | 51.17% | 75.37% | 100.00% |
| ragtruth | 2700 | 64.26% | 78.26% | 100.00% |
| rjudge | 571 | 42.73% | 87.22% | 100.00% |
| routerbench_0shot_cost_aware | 2500 | 16.84% | 16.56% | 100.00% |
| routerbench_0shot_quality | 2500 | 68.44% | 80.36% | 100.00% |
| routerbench_5shot_cost_aware | 2500 | 10.16% | 13.20% | 100.00% |
| routerbench_5shot_quality | 2500 | 77.20% | 83.44% | 100.00% |
| tsbench_agentdojo-traj | 1220 | 66.89% | 93.36% | 89.84% |
| tsbench_agentharm-traj | 731 | 28.73% | 43.64% | 100.00% |
| tsbench_asb-traj | 5231 | 51.23% | 56.60% | 100.00% |
| when2call | 3652 | 38.31% | 81.82% | 100.00% |

Classification rows show accuracy with unsupported inputs counted wrong. RouterBench rows show tie-aware oracle-optimal rates; realized quality, cost, utility and regret in [the metrics](progress.json) are the primary routing outcomes. Do not interpret these heterogeneous rows as a single general-quality index.

Causal Gemma has 26,072 valid predictions and 124 AgentDojo inputs over its 8,192-token limit. Jev returned valid answers for all 26,196 cases. No input was truncated or candidate list reduced. The 2,957 Jev predictions lost to the fixed collector persistence bug were re-requested; recovery did not select by correctness.

The earlier pilot lead does not carry over to these broader classification tasks. This run does not isolate the cause of the gap or establish what further training will achieve.

Model checkpoints, temperatures and strict math SDPA are frozen. This evaluation extends Gemma beyond its release validation envelope to 8,192 tokens and eleven options. Public-data contamination is unknown; RouterBench is an initial adapted subset and safety tasks are static judgments, not end-to-end agent benchmarks. See [the frozen protocol](../../PUBLIC_EVAL_PROTOCOL.md).

Private checkpoint snapshot: `12cf2de46fe01eee3a873a2c8f51ff84faf4b983` in [the output dataset](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval). Per-case predictions and raw inputs are not published here.
