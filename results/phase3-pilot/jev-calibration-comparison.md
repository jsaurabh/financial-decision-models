# Matched Jev calibration diagnostic

Post-training diagnostic on the reported calibration cases; not a reserved final test. No training or threshold fitting.

| Task | Cases | Gemma | Jev | Gemma minus Jev |
|---|---:|---:|---:|---:|
| banking_intent | 100 | 91.0% | 97.0% | -6.0 pp |
| contract_entailment | 100 | 85.0% | 72.0% | +13.0 pp |
| evidence_grounding | 100 | 72.0% | 82.0% | -10.0 pp |
| malicious_request | 100 | 95.0% | 85.0% | +10.0 pp |
| prompt_injection | 100 | 93.0% | 56.0% | +37.0 pp |
| tool_action_safety | 100 | 61.0% | 55.0% | +6.0 pp |
| tool_response_preference | 100 | 98.0% | 91.0% | +7.0 pp |

Resolved providers: `{"jev-1.14.0": 700}`.

Identical source inputs, question text, candidate order and reference labels; input hashes verified. Gemma is the development-selected step1200 artifact with its saved temperature. Reference labels never enter Jev requests.

## Limits

- 100 decisions per task; source-group independence is smaller than the row count.
- Native source labels have not been independently expert-adjudicated for this comparison. When2Call preferences are synthetic source supervision, not the native four-way action test; banking uses the earlier eight-candidate adaptation.
- Gemma calibration metrics are from the recorded frozen training artifact, not a new GPU replay.
- No per-case Gemma outputs were saved by the trainer; paired significance/discordance is not computed.
- Temperature was fitted on this fold for Gemma. We compare top-1 decisions, which positive scalar temperature does not change.
- This comparison is not an untouched final-test or production-quality claim.

See [metrics and pinned provenance](jev-calibration-comparison.json). Per-case Jev probabilities stay private.
