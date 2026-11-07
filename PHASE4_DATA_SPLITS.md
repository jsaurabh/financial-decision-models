# Phase4 dataset splits

All nine core tasks appear in training, validation, calibration and core test. Both candidates start from fresh official instruction bases; neither inherits a prior project classifier. Test files are separate from the training archive.

| Core task | Source | Train | Validation | Calibration | Core test |
|---|---|---:|---:|---:|---:|
| financial_verification | TAT-QA numerical verification | 5000 | 400 | 200 | 600 |
| banking_intent | BANKING77, full77 classes | 6000 | 300 | 150 | 500 |
| legal_maud | MAUD, whole-contract split | 4000 | 500 | 250 | 700 |
| legal_topic | LEDGAR, full100 classes | 5000 | 250 | 125 | 350 |
| routing_quality | RouterBench11-model quality outcomes | 4337 | 298 | 198 | 440 |
| routing_costaware | RouterBench11-model cost-aware outcomes | 4337 | 298 | 198 | 440 |
| tool_action_safety | TS-Bench native three-way harm | 1600 | 200 | 100 | 310 |
| prompt_injection | TS-Bench native interaction injection | 1600 | 200 | 100 | 310 |
| malicious_request | TS-Bench native malicious-request label | 1600 | 200 | 100 | 310 |
| **Total** | | 33474 | 2646 | 1421 | 3960 |

## Separate source/task transfer

| Domain | Heldout source(s) | Decisions |
|---|---|---:|
| financial | FinQA | 360 |
| legal | LegalBench clauses/insurance | 498 |
| model_routing | RouterArena3-model replay | 236 |
| tool_safety | TraceSafe binary/five-way risks | 520 |

Transfer total: **1614 decisions** in **617 source families**. Core and transfer scores will be reported separately.

MAUD native files overlap contracts and were explicitly repartitioned. TS-Bench supervised train+validation was split by whole request family so all three native capabilities occur in every core fold. Other source partitions and exact/near context collisions are detailed in the manifest.

Historical exposure: Some native benchmark core-test examples were evaluated by earlier models. They are excluded from this cycle training/selection, but are not claimed never inspected by the project.

These published transfer sources/cases were evaluated by earlier models; they are evaluation-only for phase4, not claimed never inspected during the project.

No Jev predictions or teacher receipts are training labels. Source-native labels and safely checked numerical counterfactuals provide supervision. Routing profiles use only retained training outcomes. No source benchmark names, future per-input outcomes or mutation metadata enter model inputs.

Some sources lack original document/report IDs; released context/text clusters are used and this limitation is retained. Public pretraining exposure is unknown. These are research benchmark tasks, not production financial-workflow validation.

[Frozen protocol](PHASE4_TRAINING_PROTOCOL.md) · [Exact source/count/hash manifest](data/phase4-manifest.json).
