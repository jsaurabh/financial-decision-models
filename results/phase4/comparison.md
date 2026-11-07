# Aligned four-domain phase4 comparison

Status: **complete**. Primary candidate: **gemma**. User specified Gemma only; checkpoint selection uses validation only.

All core tasks were represented in training, validation, calibration and core test. Source/task transfer is separate and never used for fitting or selection.

The Qwen training arm was cancelled after the user clarified the Gemma-only focus, before any phase4 Qwen test evaluation. It is excluded from this comparison. See the [scope addendum](../../PHASE4_SCOPE_ADDENDUM.md).

## core

| Task | Requested | Gemma accuracy | Jev accuracy | Coverage Gemma / Jev |
|---|---:|---:|---:|---|
| banking_intent | 500 | 76.8% | 83.6% | 100.0% / 100.0% |
| financial_verification | 600 | 53.2% | 91.5% | 100.0% / 100.0% |
| legal_maud | 700 | 80.7% | 76.3% | 100.0% / 100.0% |
| legal_topic | 350 | 70.0% | 78.3% | 100.0% / 100.0% |
| malicious_request | 310 | 96.8% | 76.1% | 100.0% / 100.0% |
| prompt_injection | 310 | 97.1% | 71.0% | 100.0% / 100.0% |
| routing_costaware | 440 | 35.2% | 10.2% | 100.0% / 100.0% |
| routing_quality | 440 | 81.1% | 80.7% | 100.0% / 100.0% |
| tool_action_safety | 310 | 81.0% | 61.0% | 100.0% / 100.0% |

Classification entries count missing decisions wrong. Routing accuracy is only tie-aware oracle-choice agreement; actual quality/cost/utility/regret and training-profile baselines are in the JSON and are the routing outcome metrics.
## transfer

| Task | Requested | Gemma accuracy | Jev accuracy | Coverage Gemma / Jev |
|---|---:|---:|---:|---|
| transfer/cuad_anti-assignment | 100 | 86.0% | 88.0% | 100.0% / 100.0% |
| transfer/cuad_audit_rights | 98 | 79.6% | 86.7% | 100.0% / 100.0% |
| transfer/cuad_cap_on_liability | 100 | 66.0% | 72.0% | 100.0% / 100.0% |
| transfer/cuad_termination_for_convenience | 100 | 90.0% | 90.0% | 100.0% / 100.0% |
| transfer/finqa_verification | 360 | 51.9% | 90.8% | 100.0% / 100.0% |
| transfer/insurance_policy_interpretation | 100 | 42.0% | 50.0% | 100.0% / 100.0% |
| transfer/routerarena_costaware | 118 | 83.9% | 84.7% | 100.0% / 100.0% |
| transfer/routerarena_quality | 118 | 78.0% | 76.3% | 100.0% / 100.0% |
| transfer/tracesafe/binary/AmbiguousArg | 20 | 0.0% | 40.0% | 80.0% / 100.0% |
| transfer/tracesafe/binary/ApiKeyLeak | 20 | 0.0% | 45.0% | 85.0% / 100.0% |
| transfer/tracesafe/binary/DataLeak | 20 | 0.0% | 100.0% | 75.0% / 100.0% |
| transfer/tracesafe/binary/DescriptionMismatch | 20 | 0.0% | 85.0% | 80.0% / 100.0% |
| transfer/tracesafe/binary/HallucinatedArgValue | 20 | 0.0% | 10.0% | 100.0% / 100.0% |
| transfer/tracesafe/binary/HallucinatedTool | 20 | 0.0% | 100.0% | 100.0% / 100.0% |
| transfer/tracesafe/binary/MissingTypeHint | 20 | 5.0% | 55.0% | 75.0% / 90.0% |
| transfer/tracesafe/binary/PromptInjectionIn | 20 | 0.0% | 55.0% | 95.0% / 100.0% |
| transfer/tracesafe/binary/PromptInjectionOut | 20 | 0.0% | 30.0% | 100.0% / 100.0% |
| transfer/tracesafe/binary/RedundantArg | 20 | 0.0% | 100.0% | 100.0% / 100.0% |
| transfer/tracesafe/binary/UserInfoLeak | 20 | 0.0% | 90.0% | 80.0% / 95.0% |
| transfer/tracesafe/binary/VersionConflict | 20 | 0.0% | 25.0% | 95.0% / 100.0% |
| transfer/tracesafe/binary/benign | 20 | 80.0% | 50.0% | 80.0% / 85.0% |
| transfer/tracesafe/coarse/AmbiguousArg | 20 | 0.0% | 30.0% | 80.0% / 100.0% |
| transfer/tracesafe/coarse/ApiKeyLeak | 20 | 5.0% | 10.0% | 85.0% / 100.0% |
| transfer/tracesafe/coarse/DataLeak | 20 | 5.0% | 50.0% | 70.0% / 100.0% |
| transfer/tracesafe/coarse/DescriptionMismatch | 20 | 0.0% | 0.0% | 75.0% / 100.0% |
| transfer/tracesafe/coarse/HallucinatedArgValue | 20 | 0.0% | 0.0% | 100.0% / 100.0% |
| transfer/tracesafe/coarse/HallucinatedTool | 20 | 0.0% | 100.0% | 100.0% / 100.0% |
| transfer/tracesafe/coarse/MissingTypeHint | 20 | 0.0% | 50.0% | 75.0% / 90.0% |
| transfer/tracesafe/coarse/PromptInjectionIn | 20 | 5.0% | 25.0% | 95.0% / 100.0% |
| transfer/tracesafe/coarse/PromptInjectionOut | 20 | 0.0% | 10.0% | 100.0% / 100.0% |
| transfer/tracesafe/coarse/RedundantArg | 20 | 0.0% | 100.0% | 100.0% / 100.0% |
| transfer/tracesafe/coarse/UserInfoLeak | 20 | 25.0% | 80.0% | 75.0% / 95.0% |
| transfer/tracesafe/coarse/VersionConflict | 20 | 0.0% | 5.0% | 95.0% / 100.0% |
| transfer/tracesafe/coarse/benign | 20 | 80.0% | 70.0% | 80.0% / 85.0% |

Classification entries count missing decisions wrong. Routing accuracy is only tie-aware oracle-choice agreement; actual quality/cost/utility/regret and training-profile baselines are in the JSON and are the routing outcome metrics.

## Claim gates

- gemma: represented-core superiority gate **not passed**. Primary effects, Holm correction, F1 retention, safety tradeoffs and coverage are in the JSON.

## Provenance and limitations

- TATQA/LEDGAR do not release original-report/document identifiers for all contexts; group on released context/request IDs plus exact/near content.
- BANKING77/LEDGAR use full native77/100 labels; MAUD uses declared eligible native review questions with complete task-specific ontologies.
- MAUD source folds overlap contracts; explicit document repartition replaces them.
- Native/public labels and programmatically checked counterfactuals are not independent production financial-workflow adjudication.
- Routing is historical released11-model outcomes, not current production routing.
- Some native benchmark core-test examples were evaluated by earlier models. They are excluded from this cycle training/selection, but are not claimed never inspected by the project.
- These published transfer sources/cases were evaluated by earlier models; they are evaluation-only for phase4, not claimed never inspected during the project.

See [full results](comparison.json), [training protocol](../../PHASE4_TRAINING_PROTOCOL.md) and [frozen manifest](../../data/phase4-manifest.json). Per-case outputs and model weights are private.
