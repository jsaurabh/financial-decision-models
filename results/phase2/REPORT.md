# Phase 2 decision quality

Both Gemma continuations improve test accuracy by about 18 percentage points over v4. Causal Gemma, joint Gemma, the Qwen reference and live Jev are within five correct cases of one another on this mixture. The attention comparison is inconclusive.

The development-selected Gemma arm is **Gemma joint bidirectional (C)**. Selection used development NLL before final test scoring. CPU inference was outside this experiment.

All five models were compared on the same 1,011 frozen decision inputs. The corpus has 6,131 training, 859 development and 851 calibration cases.

| Model | Test accuracy | Domain macro accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Previous Gemma v4 | 73.29% | 62.11% | 0.5994 | 0.3338 | 0.0556 |
| Gemma causal continuation (A) | 91.30% | 90.45% | 0.1760 | 0.1014 | 0.0155 |
| Gemma joint bidirectional (C) | 91.10% | 90.80% | 0.1945 | 0.1107 | 0.0146 |
| Qwen 27B reference | 90.90% | 89.44% | 0.2317 | 0.1260 | 0.0335 |
| Live Jev | 90.80% | 88.97% | 0.3828 | 0.1273 | 0.0279 |

Local models received a temperature fitted on the separate calibration split; Jev uses its native API probabilities. ECE and NLL therefore include this domain calibration advantage for local models. Accuracy is unaffected by a positive scalar temperature. Candidate-position macro-F1 is an implementation diagnostic; per-domain semantic macro-F1 is in the JSON reports.

| Domain | Cases | v4 | A | C | Qwen | Jev |
|---|---:|---:|---:|---:|---:|---:|
| banking intent | 213 | 95.77% | 96.71% | 96.24% | 96.24% | 96.24% |
| capability routing | 66 | 80.30% | 100.00% | 100.00% | 100.00% | 100.00% |
| concentration arithmetic | 67 | 38.81% | 100.00% | 97.01% | 92.54% | 85.07% |
| contract exception | 69 | 42.03% | 100.00% | 100.00% | 100.00% | 100.00% |
| deadline arithmetic | 72 | 44.44% | 62.50% | 72.22% | 66.67% | 68.06% |
| evidence precedence | 72 | 47.22% | 100.00% | 100.00% | 100.00% | 100.00% |
| legal contract | 291 | 90.03% | 90.38% | 88.66% | 92.10% | 92.78% |
| legal terms | 89 | 84.27% | 97.75% | 97.75% | 92.13% | 93.26% |
| net exposure arithmetic | 72 | 36.11% | 66.67% | 65.28% | 65.28% | 65.28% |

C minus A accuracy: **-0.20 percentage points**, family-cluster bootstrap 95% interval [-1.50, +1.15]. A alone was correct on 27 cases; C alone on 25. This is one paired seed and a continuation from a causal checkpoint; it does not establish a general causal-mask advantage.

The numerical-rule curriculum was useful, but exact net-exposure arithmetic and working-day counting remain the main weaknesses. A mask change alone does not resolve them. Generated routing and simple exception-rule success should not be read as broad financial expertise.

## Training and release evidence

Both arms consumed the same **2,640,467 packed tokens** in 1,000 updates. Training-stream SHA-256: `4c6ffedf6b99e9c3def3ef2cf622a0486282c7315f591bb6b566170b7a661fe0`.

| Arm | Selected step | Initial development NLL | Selected development NLL | Final calibration temperature |
|---|---:|---:|---:|---:|
| causal | 800 | 0.8327 | 0.2110 | 2.2241 |
| joint | 800 | 0.7573 | 0.2050 | 2.1483 |

## Post-test numerical diagnostic

A 30-input audit used 27 fixed development examples and three already inspected failures. It does not select a checkpoint or establish a new quality score. The strict math backend disables Flash, memory-efficient and cuDNN SDPA. FP32 upcasts the existing base weights while retaining adapter/readout values; there is no retraining.

| Precision | Attention backend | A reorder flips | C reorder flips | Largest probability change |
|---|---|---:|---:|---:|
| BF16 | native | 1 | 2 | 0.564 |
| BF16 | math | 0 | 0 | 0.00997 |
| FP32 | native | 1 | 1 | 0.569 |
| FP32 | math | 0 | 0 | 1.16e-06 |

The native backend remains sensitive with FP32 tensors. Strict math SDPA eliminates prediction flips on this audit in both precisions; FP32 plus math also reduces probability differences to about 1e-6. This supports testing a consistent backend throughout the next training/evaluation round. It does not replace the frozen native-backend results above. See [raw diagnostic](precision-audit/summary.json).

Artifacts are private Hugging Face adapters over the pinned official Gemma base; load with `finance_decisions.model.DecisionModel.load`. The saved config selects the attention mode. The previous release remains available.

- [Causal model](https://huggingface.co/jsaurabh/gemma4-finance-phase2-causal)
- [Joint model](https://huggingface.co/jsaurabh/gemma4-finance-phase2-joint)
- [Sealed corpus](https://huggingface.co/datasets/jsaurabh/financial-decision-phase2)
- [Frozen protocol](../../PHASE2_PROTOCOL.md)
- [Full paired results](comparison.json)

## Limits

Candidate-reordering agreement is 99.60% for A and 99.11% for C on this test. A few probability changes are large, including a high-confidence C flip; these are robustness failures, not established rounding-only effects. See `order-audit.json`. Tiny native-model tests establish the intended mask symmetry, but do not establish numerical stability of the trained full-size checkpoint.

Training enabled TF32 matmul; final inference uses the recorded PyTorch defaults. Training selection records remain authoritative. The root release temperature is refitted on calibration in the final inference runtime; no test labels select it.

This is a public-data and generated-rule pilot, not production financial validation. BANKING77/LEDGAR use eight sampled candidates including the reference. TOS clauses are stratified rather than sampled at deployment prevalence. Generated numerical and contract policies are hypothetical and not human reviewed. Routing labels express declared capabilities, not measured vendor utility. Parameter families and registers are held out, but logical generators are shared. Pretraining contamination cannot be ruled out; Qwen also has different earlier training data. The hard-case curriculum was not separately ablated. New test results are now inspected and should not become a future untouched holdout.

See per-model summaries for calibration, order agreement, domain metrics and exact model/runtime provenance.
