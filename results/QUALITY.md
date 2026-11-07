# Gemma 4 E2B vs live Jev: quality report

The later [phase-2 report](phase2/REPORT.md) compares matched attention continuations and a larger Qwen reference on newly sealed cases. The results below preserve the earlier v4 benchmark.

V4 uses rank-16 LoRA, a shared scalar candidate scorer, independent causal candidate blocks, and temperature calibration after merging the saved weights. Training used Colab G4 (RTX PRO 6000 Blackwell Server Edition). CPU optimization is deferred.

## Fresh public holdout

1,419 previously uninspected public test examples: 426 banking intents, 593 contract topics, and 400 legal-term predicates. The holdout covers 71 banking intents and 100 contract topics, excludes prior rows and normalized states, and was sealed before the V4 run. It was not used for training, calibration, checkpoint selection, or architecture changes after seeing these results.

Banking and contract tasks use eight sampled candidates including the reference. These are not full 77/100-class benchmarks. The legal-term task asks a binary predicate about one clause type.

| Task | Cases | Gemma v4 accuracy | Jev 1.14 accuracy |
|---|---:|---:|---:|
| Banking intent | 426 | 94.37% | 93.66% |
| Contract topic | 593 | 90.89% | 93.25% |
| Legal-term predicate | 400 | 98.75% | 95.25% |
| Overall | 1419 | 94.15% | 93.94% |

The models are effectively tied on this particular mixture. Gemma gets 47 cases uniquely correct and Jev 44; both are correct on 1,289 and wrong on 39. A three-case difference does not establish general superiority. Public-data pretraining contamination cannot be excluded.

## Binary operating points

The 400 legal-term predicates contain 20 positives (5%). Both models use their default 0.5 binary decision threshold; no threshold is tuned on this holdout.

| Metric | Gemma | Jev |
|---|---:|---:|
| Positive precision | 89.47% | 51.28% |
| Positive recall | 85.00% | 100.00% |
| Positive f1 | 87.18% | 67.80% |
| Positive average precision | 88.65% | 92.90% |

Gemma makes fewer false-positive decisions at this operating point. Jev has higher average precision, so the hard-decision F1 result should not be interpreted as superiority across all thresholds.

The subsequent [precision–recall analysis](operating-points/README.md) reports held-out diagnostic maximum F1 of 87.2% for Gemma and 85.7% for Jev. This much narrower gap reinforces the need for task-specific operating points. These test-set sweeps do not select deployed thresholds or change the frozen V4 result.

## Remaining gaps

| Probe | Unique matched inputs | Gemma | Jev |
|---|---:|---:|---:|
| Inspected domain test mix | 1712 | 93.81% | 97.31% |
| Earlier banking rules | 103 | 84.47% | 97.09% |
| Synthetic policy paraphrases | 183 | 73.22% | 100.00% |

Routing accuracy is 75.00% for Gemma versus 99.06% for Jev. The routing probe repeats four held-out task phrasings, so it is a narrow capability-routing check, not an empirical production model router.

Numeric comparison accuracy on unique earlier banking cases is 40.74% versus 88.89%. Exact arithmetic and unfamiliar policy wording remain weaknesses. These probes have informed development and are not independent final quality estimates. The original 128-case protocol, before deduplication, scores Gemma 86.72% and Jev 97.66%.

## Stability, calibration, and verification

- Candidate-reordering agreement on the fresh public holdout: 99.93%. Two runs differ on one near-boundary case; finite-precision kernels can change ties even with order-independent computation.
- Temperature is fitted on a separate calibration split after merging the release weights. Confidence remains conditional on the supplied candidates.
- At a maximum-probability threshold of 0.9, the observed fresh-holdout coverage is 83.58% with 98.06% accuracy on retained cases. This is a descriptive holdout result, not a validated production threshold.
- Seven unit tests pass, including native Gemma masking/padding/permutation and gradient checks, target-free Jev request construction, and strict comparison joins.
- The saved V4 artifact passes an in-process HTTP API smoke test for choice, noul, score, and invalid-schema rejection.
- All 4,219 live Jev benchmark calls succeeded and resolved to `jev-1.14.0`. Requests contain state and question only; reference labels are joined after inference.
- Matched comparisons require identical input hashes, candidate order, and references. Repeated semantic inputs are deduplicated; original case-weighted results remain available.
- The private Hugging Face weight revision and SHA-256 are recorded in `gemma4-finance-v4/artifact.json`. The G4 runtime was released after reports and weights were saved.

Jev probability metrics use its returned probabilities, clipped at 1e-12 for log loss. Provider-side rounding can affect NLL. Hosted network latency is not compared with local GPU or CPU inference.

## Artifacts

- [Private model](https://huggingface.co/jsaurabh/gemma4-e2b-finance-decisions-v4)
- [Code PR](https://github.com/jsaurabh/financial-decision-models/pull/1)
- [Complete paired metrics](gemma-vs-jev.json)
- [Fresh holdout predictions](gemma4-finance-v4/public-fresh-predictions.jsonl)
- [Saved-artifact API check](gemma4-finance-v4/api-smoke.json)

The next quality work should prioritize diverse reviewed routing cases and arithmetic/policy generalization. These public and synthetic results do not validate production financial decisions.
