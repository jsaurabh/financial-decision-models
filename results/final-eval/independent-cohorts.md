# Completed pre-TraceSafe independent cohorts

**The frozen phase-3 Gemma pilot trails Jev on all four independent classification domains.** Equal-domain accuracy is52.0% versus83.3%; Gemma minus Jev is−31.3 percentage points (paired family95% interval −34.0 to−28.6). The prespecified broader-classification-superiority gate is not passed.

All11,634 independent classification decisions are complete with100% prediction coverage for both systems. All secondary routing/attack-recall decisions also have receipts; Gemma has four unsupported routing decisions. The separate26,196-case development-informed regression replay was still running when this snapshot was generated and is excluded here. TraceSafe is not included.

| Independent domain | Decisions | Gemma accuracy | Jev accuracy | Difference |
|---|---:|---:|---:|---:|
| Legal (equal mean of five tasks) | 4,111 | 48.8% | 77.2% | -28.4 pp |
| CLINC150 out-of-scope rejection | 5,497 | 67.8% | 90.7% | -22.8 pp |
| FinQA answer verification | 1,910 | 50.2% | 89.5% | -39.3 pp |
| Prompt-injection text detection | 116 | 41.4% | 75.9% | -34.5 pp |
| Equal-domain average | 11,634 | 52.0% | 83.3% | -31.3 pp |

The average gives each of the four domains equal weight; legal gives its five tasks equal weight. It is a prespecified research mixture, not production traffic weighting. All four paired domain comparisons favor Jev, with Holm-adjusted p=0.0004. Public pretraining exposure remains unknown.

## Restricted RouterArena cached replay

Both routing objectives use the same fixed three-model pool and809 native requests. Gemma supports807/809 requests per objective; Jev supports809/809. Quality/cost/utility comparisons below use807 common-valid requests. They are separate from classification accuracy.

| Objective | Gemma selected quality | Jev selected quality | Gemma USD/request | Jev USD/request |
|---|---:|---:|---:|---:|
| routerarena_quality | 0.6597 | 0.6120 | 0.00006720 | 0.00008756 |
| routerarena_costaware | 0.6831 | 0.6785 | 0.00005692 | 0.00005050 |

For quality-focused routing, Gemma gains0.0478 released quality points (paired family95% interval0.0249 to0.0706) and selects models with approximately23.3% lower cached cost. This supports a narrow positive result on this historical restricted-pool replay. It does not establish full RouterArena or production-routing superiority.

For cost-aware routing, the utility difference is+0.0046 (95% interval−0.0111 to0.0198), so superiority is not established.

## Calibration context

The earlier700-case calibration diagnostic averaged85.0% for Gemma versus76.9% for Jev across seven equally sized tasks. That source-family diagnostic is not the independent final test; its advantage did not transfer to these new classification tasks.

The62 eligible InjecAgent attacks are secondary, all-positive and share one uncertainty family: Gemma recall0.0%, Jev80.6%. They cannot establish benign false-block performance or general tool safety.

## Provenance and limits

- Frozen Gemma artifact: `jsaurabh/gemma4-finance-phase3-pilot@1eec2a39d4d6a3106d5175b73539293ae3259cf0`, selected step1200 with its saved temperature. Jev uses the frozen alias/version requirement from the manifest.
- No model, temperature, prompt, threshold, reference label or eligibility rule was changed after final predictions began.
- Legal and OOS are public classification tests; FinQA is a declared numerical-verification adaptation. Results do not establish production financial-workflow quality.
- Routing is a historical three-model replay with released native quality/cost outcomes. Over-context decisions stay in the requested oracle-optimal accuracy denominator and are disclosed separately from common-valid utility.
- Full-regression results will be published separately by the original background collector. This completed-cohort snapshot will remain unchanged.

See [full paired metrics](independent-cohorts.json), [frozen protocol](../../FINAL_EVAL_PROTOCOL.md) and [source/count/hash manifest](../../data/final-eval-manifest.json). Raw contexts and per-case predictions remain private.
