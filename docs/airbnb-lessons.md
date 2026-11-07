# Applying the Airbnb classifier benchmark

Airbnb reports a narrow production policy task, hundreds of thousands of in-domain training examples, and a 10,000-example evaluation weighted to correct its sampling scheme. It emphasizes precision–recall operating points. Its reported 35 ms median and 85 ms p99 use GPU serving, so they do not establish a CPU target for this project. Source: [Airbnb Engineering](https://airbnb.tech/ai-ml/frontier-quality-at-1-of-the-cost-a-production-benchmark-of-llm-classifiers/).

The useful transfer is a more specific quality objective and better data. The article does not disclose enough architecture or training detail to reproduce its exact recipe.

| Project change | Why it matters here | Status |
|---|---|---|
| Separate stable workflows and label policies | Banking intents, contract topics, legal predicates, and model routing have different error costs. A pooled accuracy score hides them. | Per-domain metrics exist; production workflow definitions and targets remain to be supplied. |
| Grow diverse in-domain examples and hard negatives | Our routing probe repeats four task phrasings, and the legal-predicate holdout has only 20 positives. More epochs on the same templates cannot establish broad quality. | Prioritize varied routing requests, ambiguous clauses, missing evidence, and counterfactual policy cases. Plot learning curves at increasing labeled-data budgets. |
| Compare score distributions and operating points | Gemma has higher default legal-predicate F1, while Jev has higher average precision. Native 0.5 decisions answer only one tradeoff. | Added complete PR curves, explicit tie handling, and maximum-F1 diagnostics. |
| Select thresholds per task on calibration data | A precision target for a legal flag differs from a recall target for a review route. A score-temperature fit alone does not choose either decision threshold. | Added calibration-tag checks, per-task fitting, dataset fingerprints, and overlap rejection when evaluating frozen policies. |
| Evaluate on a realistic traffic mix | Our public holdout is a designed benchmark mix, not observed financial-services traffic. | Weighted binary metrics are supported; weights must come from known sampling probabilities. No traffic weights are invented. |
| Keep serving inputs compact after quality is established | Fixed workflows can reduce repeated schema text. A fixed-taxonomy head is a useful future ablation against the current dynamic candidate scorer. | CPU experiments remain deferred. Future checks should measure p50/p95/p99 under load, throughput, and post-quantization quality on the target hardware. |

For exact numeric policies, a separate lesson from our own errors is to compute arithmetic and explicit comparisons in deterministic code where possible. This is a proposed workflow design change, not a modification to the matched-input Jev benchmark or a claim made by Airbnb.

## Concrete additions in this update

The [operating-point report](../results/operating-points/README.md) uses existing matched predictions. No GPU was allocated, no model weights changed, and no new provider calls were made.

At the default decision point, legal-predicate F1 is 87.2% for Gemma and 67.8% for Jev. A held-out threshold sweep gives descriptive maxima of 87.2% and 85.7%. These maxima inspect evaluation labels and are optimistic diagnostics; they are not selected production thresholds. Jev's average precision is higher: 92.9% versus 88.7%.

For the next evaluation run, `--export-calibration` saves explicitly tagged calibration predictions and stable task keys:

```bash
python -m finance_decisions.evaluate_quality \
  --artifact runs/gemma4-finance-v4/release \
  --out results/next-evaluation \
  --hf-repo jsaurabh/gemma4-e2b-finance-decisions-v4 \
  --export-calibration --public-holdout
```

Filter the calibration records to one binary `task_key`, choose the objective from the workflow's error costs, and fit the policy:

```python
from finance_decisions.operating_points import fit_threshold, evaluate_frozen_threshold

# Example objective only: the workflow owner must set the precision requirement.
policy = fit_threshold(task_calibration_records, min_precision=0.90,
                       min_predicted_positives=20)
if policy['status'] == 'selected':
    metrics = evaluate_frozen_threshold(policy, independent_task_records,
                                        task_calibration_records)
```

An unattainable constraint returns no threshold. A selected threshold reports its empirical calibration support; it does not guarantee future precision. Freeze the model, score calibrator, and decision threshold as one versioned policy, and preserve it during independent evaluation. Rare-event precision requirements need substantially more positive and predicted-positive examples than this small public slice provides.
