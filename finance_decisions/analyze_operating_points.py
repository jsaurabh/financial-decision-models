"""Diagnose frozen predictions without selecting or deploying a test-set policy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'finance-operating-points'

from .compare_quality import pair_records
from .operating_points import binary_curve, threshold_metrics


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gemma', default='results/gemma4-finance-v4/public-fresh-predictions.jsonl')
    parser.add_argument('--jev', default='results/jev-live/public-fresh-predictions.jsonl')
    parser.add_argument('--domain', default='legal_terms')
    parser.add_argument('--out', default='results/operating-points')
    args = parser.parse_args()
    left = [r for r in load(args.gemma) if r['domain'] == args.domain]
    right = [r for r in load(args.jev) if r['domain'] == args.domain]
    left, right = pair_records(left, right)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = {'domain': args.domain, 'evaluation_only': True,
              'threshold_policy_selected': False,
              'note': 'Maximum F1 and precision-constrained points are optimistic held-out diagnostics. Never deploy these thresholds; select on separate calibration data.',
              'models': {}}
    figure, axis = plt.subplots(figsize=(7, 4.6))
    for name, records, color in [('Gemma v4', left, '#2463a5'), ('Jev 1.14', right, '#c36216')]:
        curve = binary_curve(records)
        if curve['average_precision'] is None:
            raise ValueError('Precision–recall diagnostics require both positive and negative examples')
        default = threshold_metrics(records, .5, inclusive=False)
        maximum = max(curve['points'], key=lambda p: (p['f1'], p['precision'] or 0))
        constrained = {}
        for minimum in [.8, .9, .95]:
            eligible = [p for p in curve['points'] if p['precision'] is not None and p['precision'] >= minimum]
            constrained[str(minimum)] = max(eligible, key=lambda p: (p['recall'], p['precision'])) if eligible else None
        result['models'][name] = {**curve, 'default_rule': 'p_true > 0.5; ties false',
            'default': default, 'heldout_oracle_max_f1': maximum,
            'heldout_oracle_precision_constraints': constrained}
        points = curve['points']
        axis.step([0.] + [p['recall'] for p in points], [1.] + [p['precision'] for p in points], where='pre',
                  label=f'{name}: AP {curve["average_precision"]:.3f}', color=color)
        axis.scatter(default['recall'], default['precision'], color=color, s=60, zorder=4)
    axis.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1.02), ylim=(0, 1.04),
             title=f'Legal-term predicates: {len(left)} cases, {sum(r["target"] == "true" for r in left)} positives')
    axis.grid(alpha=.2); axis.legend(loc='lower left')
    figure.text(.5, .01, 'Dots: native 0.5 operating points. Curves are held-out diagnostics, not deployed thresholds.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0,.035,1,1))
    svg_path = out / 'precision-recall.svg'
    figure.savefig(svg_path, metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    figure.savefig(out / 'precision-recall.png', dpi=180)
    plt.close(figure)
    (out / 'diagnostics.json').write_text(json.dumps(result, indent=2))
    lines = ['# Precision–recall operating points', '',
             'This analysis incorporates the score-thresholding lesson from [Airbnb’s classifier benchmark](https://airbnb.tech/ai-ml/frontier-quality-at-1-of-the-cost-a-production-benchmark-of-llm-classifiers/). It reuses saved matched predictions; no new training or API calls are required.', '',
             f'Legal-term predicates: {len(left)} cases, {sum(r["target"] == "true" for r in left)} positives. The questions cover several clause types; this pooled diagnostic is not a task-specific production threshold.', '',
             '![Precision–recall curves](precision-recall.svg)', '',
             '| Model | Default precision | Default recall | Default F1 | Average precision | Held-out maximum F1* |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in result['models'].items():
        point = values['default']
        lines.append(f'| {name} | {point["precision"]:.1%} | {point["recall"]:.1%} | {point["f1"]:.1%} | {values["average_precision"]:.1%} | {values["heldout_oracle_max_f1"]["f1"]:.1%} |')
    lines += ['', '*Maximum F1 is a descriptive upper bound obtained by inspecting this held-out set. It is optimistic and is not a deployable threshold or an independent estimate after threshold tuning.', '',
              'The higher default F1 and accuracy for Gemma do not imply better ranking across operating points. Jev has higher average precision here. Its rounded returned scores create ties, which move together in the curve. Only 20 positives are present, so precision/recall estimates remain sensitive to individual cases.', '',
              '## Deployment protocol', '',
              '1. Define a stable task, label policy, and acceptable false-positive/false-negative tradeoff.',
              '2. Collect separate calibration predictions with explicit task keys and split labels.',
              '3. Fit each task threshold on calibration data with `fit_threshold`; freeze the model, score calibrator, data fingerprint, and threshold together.',
              '4. Apply it unchanged to an independent evaluation set with `evaluate_frozen_threshold`.',
              '5. Use production sample weights only when their sampling probabilities are known; do not infer traffic weights from this benchmark.', '',
              'No threshold or served model changed as a result of this diagnostic.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[8:14]))


if __name__ == '__main__':
    main()
