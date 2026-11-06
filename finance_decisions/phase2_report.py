"""Render the completed pilot with measured results and practical limits."""
import json
from pathlib import Path


def main():
    root=Path('results/phase2');c=json.loads((root/'comparison.json').read_text())
    names={'v4':'Previous Gemma v4','causal':'Gemma causal continuation (A)','joint':'Gemma joint bidirectional (C)','qwen':'Qwen 27B reference','jev':'Live Jev'}
    manifest=json.loads(Path('data/phase2-manifest.json').read_text())
    selection={n:json.loads((root/n/'selection.json').read_text()) for n in ['causal','joint']}
    matched=all(selection['causal'][k]==selection['joint'][k] for k in ['training_input_stream_sha256','training_tokens'])
    if not matched:raise ValueError('The attention arms did not receive an identical training stream')
    selected=c['gemma_selection']['selected'];pair=c['paired']['causal_to_joint'];lo,hi=pair['family_bootstrap_95pct']
    lines=['# Phase 2 decision quality','',f"The development-selected Gemma arm is **{names[selected]}**. Selection used development NLL before final test scoring. CPU inference was outside this experiment.",'',
        f"All five models were compared on the same {manifest['splits']['test']['rows']:,} frozen decision inputs. The corpus has {manifest['splits']['train']['rows']:,} training, {manifest['splits']['development']['rows']:,} development and {manifest['splits']['calibration']['rows']:,} calibration cases.",'',
        '| Model | Test accuracy | Domain macro accuracy | NLL | Brier | ECE |','|---|---:|---:|---:|---:|---:|']
    for n,label in names.items():
        m=c[n];lines.append(f"| {label} | {m['accuracy']:.2%} | {m['domain_macro_accuracy']:.2%} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece_10bin']:.4f} |")
    lines+=['','Local models received a temperature fitted on the separate calibration split; Jev uses its native API probabilities. ECE and NLL therefore include this domain calibration advantage for local models. Accuracy is unaffected by a positive scalar temperature. Candidate-position macro-F1 is an implementation diagnostic; per-domain semantic macro-F1 is in the JSON reports.','',
        '| Domain | Cases | v4 | A | C | Qwen | Jev |','|---|---:|---:|---:|---:|---:|---:|']
    for d in c['joint']['domains']:
        ds=[c[n]['domains'][d] for n in names];lines.append('| '+d.replace('_',' ')+' | '+str(ds[0]['n'])+' | '+' | '.join(f"{x['accuracy']:.2%}" for x in ds)+' |')
    lines+=['',f"C minus A accuracy: **{pair['b_minus_a_accuracy']*100:+.2f} percentage points**, family-cluster bootstrap 95% interval [{lo*100:+.2f}, {hi*100:+.2f}]. A alone was correct on {pair['a_only_correct']} cases; C alone on {pair['b_only_correct']}. This is one paired seed and a continuation from a causal checkpoint; it does not establish a general causal-mask advantage.",'',
        '## Training and release evidence','',f"Both arms consumed the same **{selection['causal']['training_tokens']:,} packed tokens** in 1,000 updates. Training-stream SHA-256: `{selection['causal']['training_input_stream_sha256']}`.",'',
        '| Arm | Selected step | Initial development NLL | Selected development NLL | Temperature |','|---|---:|---:|---:|---:|']
    for n in ['causal','joint']:
        s=selection[n];lines.append(f"| {n} | {s['selected_step']} | {s['initial_dev']['nll']:.4f} | {s['selected_dev_nll']:.4f} | {s['temperature']:.4f} |")
    lines+=['','Artifacts are private Hugging Face adapters over the pinned official Gemma base; load with `finance_decisions.model.DecisionModel.load`. The saved config selects the attention mode. The previous release remains available.','',
        '- [Causal model](https://huggingface.co/jsaurabh/gemma4-finance-phase2-causal)','- [Joint model](https://huggingface.co/jsaurabh/gemma4-finance-phase2-joint)','- [Sealed corpus](https://huggingface.co/datasets/jsaurabh/financial-decision-phase2)','- [Frozen protocol](../../PHASE2_PROTOCOL.md)','- [Full paired results](comparison.json)','','## Limits','',
        'This is a public-data and generated-rule pilot, not production financial validation. BANKING77/LEDGAR use eight sampled candidates including the reference. TOS clauses are stratified rather than sampled at deployment prevalence. Generated numerical and contract policies are hypothetical and not human reviewed. Routing labels express declared capabilities, not measured vendor utility. Parameter families and registers are held out, but logical generators are shared. Pretraining contamination cannot be ruled out; Qwen also has different earlier training data. The hard-case curriculum was not separately ablated. New test results are now inspected and should not become a future untouched holdout.','','See per-model summaries for calibration, order agreement, domain metrics and exact model/runtime provenance.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(root/'REPORT.md')

if __name__=='__main__':main()
