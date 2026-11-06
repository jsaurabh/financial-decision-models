# Phase 2: decision quality

This protocol is frozen before training or scoring the new test set. CPU speed and model size are out of scope. Use Colab G4; retain private model artifacts on Hugging Face.

## Matched Gemma comparison

Both arms continue from `jsaurabh/gemma4-e2b-finance-decisions-v4` revision `1307ef28b41b4b295c204f5f60f57f92caa09bb1`, path `run/selected`. The official base and tokenizer are pinned in `finance_decisions/model.py`.

- A: existing independent candidate branches, causal global and sliding layers.
- C: joint bidirectional global layers across the state, question and every candidate. Sliding layers retain A's independent causal mask.

Same initial LoRA weights and scalar readout, candidate serialization and reset positions, seed 20261108, optimizer, ordered batches, dropout seed, 2,048-token cap and 1,000 updates. Each update has eight public and eight generated hard cases, microbatch four, accumulation four. Compare recorded training-stream hashes and token counts. No silent truncation. Checkpoints at steps 0/200/400/600/800/1000; select lowest uncalibrated development NLL. Fit one positive scalar temperature per model on calibration only. A/C changes global direction and candidate connectivity together, so it cannot isolate those mechanisms. Starting from a causal-trained checkpoint may favor A; this is a continuation study, not a from-scratch architecture verdict. One paired seed is diagnostic, not a multi-seed result.

## Data and evaluation

`data/phase2-manifest.json` commits split hashes and shared eligibility before scoring. Reproduce with `build_data`, `scripts.prepare_legacy_eval`, `build_public_holdout`, `phase2_data`, then `phase2_preflight`. The earlier tests and training/calibration inputs are exclusion-only. Development, calibration and test share no normalized state or family with each other or historical cases. Exact duplicate decision inputs are removed within each split. Public test cases use remaining official test rows; development/calibration use remaining validation rows, or unused training rows for BANKING77.

New generated cases cover net exposure arithmetic, working-day deadlines, concentration ratios, contract exceptions, evidence precedence, and capability routing with distracting background text. Gold labels come from checked deterministic oracles. Synthetic registers and parameter families are split separately, but logical generators are shared; this remains a controlled generalization test, not independent production validation. Numerical policies are invented. Public pretraining contamination is unknown. BANKING77/LEDGAR have eight sampled candidates including the reference. TOS clauses are stratified, not deployment prevalence. Routing targets declared capabilities, not empirically measured model utility.

Train A and C without opening test scores. Then evaluate v4 without continuation, A, C and the larger Qwen reference on identical frozen inputs. Recalibrate every local model on the same calibration split. Qwen is pinned to `jsaurabh/qwen-decision-27b-noncausal-lora@b2eefb929a883136b3bd56ce70999fdac656ee8a`, base `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its different model size and earlier training data make it a quality reference, not part of the mask ablation. Record runtime versions separately.

Evaluate live Jev on the frozen test inputs, record the resolved provider version and failures, and send only state/question, never reference labels. Jev supplies no training labels. Preserve exact candidate order and canonical input hashes for all comparisons.

Report overall/per-domain accuracy, semantic macro-F1, NLL, Brier score, ECE, confidence/coverage, and shuffled-option agreement. Report paired A/C differences with family-cluster bootstrap intervals and exact paired discordance counts; public/source contamination and related generated templates limit statistical interpretation. Choose the Gemma arm by development NLL before seeing test scores; report Qwen separately and judge the next training direction using development results. A hard-case curriculum ablation is not included, so improvements over v4 cannot be attributed to any one data family.

No reinforcement learning is planned in this round: first establish which representation and supervised curriculum work. Do not merge or replace the previous release based only on a small aggregate test difference.
