# TraceSafe comparison

This is a separate addendum to the frozen independent final evaluation. The original corpus, historical access-block record and four-domain claim gates remain unchanged. See the [prospectively frozen protocol](../../TRACESAFE_EVAL_PROTOCOL.md) and [source/count/hash manifest](../../data/tracesafe-eval-manifest.json).

The addendum contains all1,170 native source traces, grouped into150 uncertainty families, with binary and five-way decisions for each (2,340 requested decisions). Reference/mutation/distractor/source-model annotations are excluded from the observed input. It is an offline, full-context decision-interface adaptation, not an exact reproduction of the paper's prompts or an interactive agent-security benchmark. No data is used for training or calibration.

The collector publishes full or transparently partial paired results as `comparison.md` and `comparison.json` in this directory after the workers terminate. Those reports retain unsupported inputs in the primary denominator and present common-valid sensitivity separately.

## Runtime observations

- Pinned-tokenizer preflight: Gemma supports1,023/1,170 binary inputs and1,015/1,170 five-way inputs within its frozen8,192-token limit. All302 over-context decisions remain requested; they are not truncated or removed.
- Jev completed2,340 original request attempts:2,278 valid receipts and62 error receipts. The receipts retain their original statuses.
- The earliest failed Jev input was replayed once solely to diagnose the failure. It returned HTTP400 with `error_type=max_tokens_exceeded`. This diagnostic did not replace the original prediction or affect any reference label, prompt, model setting, eligibility rule or claim gate. The other61 error receipts were not individually diagnosed.
- The diagnostic request and its input hash are retained privately at `jsaurabh/financial-decision-tracesafe-eval/diagnostics/jev-request-limit.json`. Raw traces and per-case predictions remain private.

The source-prevalence always-unsafe baseline is92.3% accuracy but only50% on the prespecified balanced primary score. Benign false blocks and coverage are required to interpret any apparent detection gain.
