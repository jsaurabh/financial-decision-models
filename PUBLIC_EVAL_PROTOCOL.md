# Frozen public evaluation protocol

Run only evaluation. Use two simultaneous Colab G4 instances if available: one for both frozen Gemma arms, one for the frozen Qwen reference. Run Jev independently. A background lifecycle controller releases each owned GPU on worker completion/failure and imposes an eight-hour maximum. No CPU optimization, fine-tuning, temperature fitting or checkpoint selection.

## Corpus

`data/public-eval-manifest.json` records counts and byte hashes, sealed before model scoring. Source preparation is `scripts/prepare_public_sources.py`, then `python -m finance_decisions.public_data`. Raw contexts remain in private working copies and are transferred directly to the user's Colab VMs, not published or uploaded to a dataset repository. GitHub keeps code/manifests/aggregate reports; private Hugging Face stores model predictions and resumable progress.

- When2Call: all 3,652 official MCQ test cases, original A–D options and published tools/question.
- TS-Bench: all 7,182 evaluation actions in the three published configurations. Model input allowlists instruction, history, current_action and env_info. Score, attacker_tool, attack_success, future outcomes and other evaluator metadata are excluded. Three-class safe/controversial/unsafe task, plus strict and loose binary diagnostics.
- R-Judge: all 571 records in the pinned repository (the paper states 569). Input includes scenario, profile and interaction contents. Risk descriptions, labels and attack annotations are excluded from inputs. Binary safety judgment, not explanation grading.
- ContractNLI: 123 official test contracts × 17 hypotheses = 2,091 decisions. Full contract retained; native three-way NLI and original hypotheses; evidence extraction is outside this run. Group by document.
- RAGTruth: all 2,700 official test responses. Match the Decision Index payload and hallucination predicate. Gold spans and quality metadata are excluded. Group by source context.
- RouterBench: 2,500 source requests per shot regime, selected by SHA-256 of a fixed seed and source ID, with both native quality and cost-aware objectives. Total 10,000 decisions. The Decision Index connected-group 20% source calibration/profile builder and eleven candidates are retained. Sampling never uses outcomes. Report each shot/objective separately; this is an initial subset, not an official whole Decision Index score. All tied optima are accepted. Report realized quality, historical cost, utility (penalty 10/USD in the cost-aware track), and oracle regret; exact agreement with an arbitrary first optimum is not the main metric.

Use the discovery source revisions and raw byte hashes. Public pretraining/upstream training contamination is unknown. Qwen has previously been evaluated on public Decision Index; familiar public tasks do not become new blind tests. No examples were selected in response to predictions.

## Models and inference

Pin artifacts from `results/phase2/artifacts.json`: Gemma causal `6b4188a4790bba0beaef19bb48e4b9b18232138d`, joint `99b0773bf266fbc4b19189fe451f5f4088021f48`, and Qwen `b2eefb929a883136b3bd56ce70999fdac656ee8a`. Retain published temperatures. Resolve and report the live Jev version.

BF16 base weights, published adapter/readout precision, strict math SDPA for local models; disable Flash, memory-efficient and cuDNN SDPA and FP32 matmul TF32. Qwen's original recurrent/linear layers remain unchanged. This backend was chosen using the earlier numerical diagnostic, before this evaluation. It differs from the earlier native-backend report.

Allow full inputs up to 8,192 model tokens; no truncation. Gemma's shared scalar head accepts an evaluation-only eleven-candidate extension and a larger character budget; the served API keeps its 8-candidate/12k-character defaults. These inference extensions exceed the release's validated 2,048-token/eight-option envelope; report them, not as a new trained release. Qwen retains its native answer-code head and prompt. Determine each model's actual token lengths independently.

Record over-context, OOM, preparation and API failures explicitly. Missing predictions count wrong for conservative accuracy; valid-only F1, routing utility and coverage appear alongside it. Compare models on identical common-valid inputs as a supplementary view, never silently drop the unsupported tail. No model receives gold labels, stored teacher predictions, outcome tables or evaluation success flags as inputs.

## Outputs and background operation

Every model writes resumable per-case predictions, pinned run configuration and per-suite coverage/metrics to the private results dataset. Checkpoint progress every 500 decisions. Save the same canonical input hashes and candidate order for paired comparisons. Raw text is excluded from prediction records. Report classification accuracy/F1, positive precision/recall/AP, safety misses/false blocks, contract document-exact match, routing outcome metrics, and confidence/coverage with task-specific denominators. Do not equate this adapted static safety test with native end-to-end agent safety.
