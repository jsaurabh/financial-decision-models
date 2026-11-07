# Gemma E2B training-recipe diagnostic

Status: **complete**. Outcome: **complete**.

All arms start from the same pinned Gemma4 E2B instruction base. Training/development examples, adapter initialization, effective batches and candidate permutations are controlled. Native SFT uses the frozen full-vocabulary LM head; code variants train the small FP32 answer-code head.

These are development comparisons. Test cases were not loaded. A higher aggregate alone does not qualify a model for deployment or establish superiority over Jev.

## Memorization checks

| Arm | Selected update | Training accuracy | Financial pairs both correct | Native top-one allowed | Passed |
|---|---:|---:|---:|---:|---|
| mem_code_lr2e-5 | 192 | 98.4% | 93.8% | n/a | True |
| mem_code_lr1e-4 | 160 | 100.0% | 100.0% | n/a | True |
| mem_sft_lr1e-4 | 160 | 95.3% | 81.2% | 100.0% | False |

The64 cases are training examples. This checks optimization and binding, not generalization.

## Matched development pilots

| Arm | Selected update | Aggregate | Financial AUC | Financial pairs both correct | Cost-aware utility | Fixed-router utility |
|---|---:|---:|---:|---:|---:|---:|
| pilot_code_logk_lr2e-5 | 384 | 0.570 | 0.490 | 0.0% | 0.581 | 0.667 |
| pilot_code_logk_lr1e-4 | 384 | 0.646 | 0.490 | 1.6% | 0.599 | 0.667 |
| pilot_code_plain_lr2e-5 | 384 | 0.568 | 0.496 | 1.6% | 0.633 | 0.667 |
| pilot_code_plain_lr1e-4 | 384 | 0.628 | 0.481 | 3.1% | 0.665 | 0.667 |
| pilot_native_sft_lr2e-5 | 384 | 0.588 | 0.500 | 4.7% | 0.680 | 0.667 |
| pilot_native_sft_lr1e-4 | 384 | 0.637 | 0.489 | 0.0% | 0.600 | 0.667 |

## Safety and decision binding

| Arm | Action unsafe/review recall | Safe false blocks | Permutation consistency | Alias consistency | Predicate-reversal consistency |
|---|---:|---:|---:|---:|---:|
| pilot_code_logk_lr2e-5 | 83.0% | 41.2% | 64.6% | 56.9% | 75.0% |
| pilot_code_logk_lr1e-4 | 93.6% | 23.5% | 71.5% | 63.9% | 56.2% |
| pilot_code_plain_lr2e-5 | 78.7% | 35.3% | 65.3% | 56.2% | 81.2% |
| pilot_code_plain_lr1e-4 | 91.5% | 23.5% | 76.4% | 72.2% | 81.2% |
| pilot_native_sft_lr2e-5 | 80.9% | 47.1% | 66.0% | 54.9% | 93.8% |
| pilot_native_sft_lr1e-4 | 89.4% | 23.5% | 71.5% | 63.2% | 0.0% |

Full histories, per-task metrics, acceptance diagnostics and immutable artifact revisions are in [comparison.json](comparison.json).
The [frozen protocol](../../data/recipe-audit-protocol.json) and [manifest](../../data/recipe-audit-manifest.json) describe the controls.

A near-chance financial AUC, low pair discrimination or utility below the fixed router remains a failure even if the aggregate rises. Preserve old benchmark receipts; subsequent work requires a separately sealed final holdout.
