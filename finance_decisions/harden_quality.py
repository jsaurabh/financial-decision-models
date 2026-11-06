"""Continue the pilot on independently split contrastive banking training families.

Selection uses development NLL only. The archived 128-case test never supplies
labels to this trainer. New checkpoints retain the same official Gemma base.
"""
import hashlib
import json
import random
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import metrics, fit_temperature
from .model import DecisionModel, counts, targets
from .train import read, filter_rows, infer, shuffled_options, report, upload
from .robustness import augment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting-artifact', default='runs/gemma4-finance-v1/selected')
    parser.add_argument('--out', default='runs/gemma4-finance-v2')
    parser.add_argument('--hf-repo', default='jsaurabh/gemma4-e2b-finance-decisions-v2')
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20261106); rng = random.Random(20261106); torch.set_num_threads(8)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = DecisionModel.load(args.starting_artifact, device='cuda')
    model.max_length = 768
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad_('lora_' in name)
    original, dropped = filter_rows(model, read('data/private/train.jsonl'))
    dev_families = {r['family'] for r in original if int(hashlib.sha256(r['family'].encode()).hexdigest()[:8], 16) % 20 == 0}
    original_train = [r for r in original if r['family'] not in dev_families]
    original_dev = [r for r in original if r['family'] in dev_families]
    gap, gap_dropped = filter_rows(model, read('data/private/contrastive-banking-train.jsonl'))
    gap_dev_families = {r['family'] for r in gap if int(hashlib.sha256(r['family'].encode()).hexdigest()[:8], 16) % 10 == 0}
    gap_dev = [r for r in gap if r['family'] in gap_dev_families]
    gap_train = [r for r in gap if r['family'] not in gap_dev_families]
    dev = original_dev + gap_dev
    if args.augment:
        dev += [augment(r, rng) for r in original_dev]
    assert not ({r['family'] for r in original_train + gap_train} & {r['family'] for r in dev})
    config = {'steps': args.steps, 'effective_batch': 16, 'max_length': model.max_length,
              'mixture': '8 original + 8 contrastive banking examples per update',
              'original_train_rows': len(original_train), 'contrastive_train_rows': len(gap_train),
              'dev_rows': len(dev), 'dropped_original': dropped, 'dropped_gap': gap_dropped,
              'starting_artifact': args.starting_artifact, 'selection': 'combined development NLL', 'paraphrase_augmentation': args.augment,
              'gpu': torch.cuda.get_device_name(), 'seed': 20261106}
    (out / 'run_config.json').write_text(json.dumps(config, indent=2)); print(json.dumps(config), flush=True)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 3e-5},
        {'params': model.readout.parameters(), 'lr': 1e-4}], weight_decay=.01)
    initial = metrics(infer(model, dev), targets(dev)); best_nll = initial['nll']; best_step = 0
    model.save(out / 'selected')
    start = time.monotonic(); log = []
    for step in range(1, args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        rows = rng.sample(original_train, 8) + rng.sample(gap_train, 8)
        if args.augment:
            rows = [augment(r, rng) if rng.random() < .8 else r for r in rows]
        rows = shuffled_options(rows, rng)
        # Separate microbatches limit memory for longer policy inputs.
        losses = []
        for offset in [0,8]:
            part = rows[offset:offset+8]
            logits = model(model.prepare(part), counts(part))
            loss = F.cross_entropy(logits, torch.tensor(targets(part), device='cuda'))
            (loss / 2).backward(); losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        entry = {'step': step, 'loss': sum(losses)/2, 'wall_s': time.monotonic() - start}
        log.append(entry)
        if step % 20 == 0:
            print(json.dumps(entry), flush=True)
        if step % 175 == 0 or step == args.steps:
            score = metrics(infer(model, dev), targets(dev))
            if score['nll'] < best_nll:
                best_nll, best_step = score['nll'], step; model.save(out / 'selected')
            print(json.dumps({'step': step, 'dev': score, 'selected_step': best_step}), flush=True)
            model.save(out / 'checkpoint')
            (out / 'training.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in log))
            upload(out / 'checkpoint', args.hf_repo, 'checkpoint')
    del model, optimizer; torch.cuda.empty_cache()
    model = DecisionModel.load(out / 'selected', device='cuda')
    calibration, _ = filter_rows(model, read('data/private/calibration.jsonl'))
    model.temperature = fit_temperature(infer(model, calibration), targets(calibration))
    test, dropped = filter_rows(model, read('data/private/test.jsonl'))
    # Calibrate the actual merged BF16 release; merging may slightly change logits.
    model.save(out / 'release', merge=True)
    model.temperature = fit_temperature(infer(model, calibration), targets(calibration))
    release_config = out / 'release' / 'decision_config.json'
    payload = json.loads(release_config.read_text()); payload['temperature'] = model.temperature
    release_config.write_text(json.dumps(payload, indent=2))
    evaluation = report(model, test, infer(model, test))
    evaluation.update({'selected_step': best_step, 'temperature': model.temperature, 'initial_dev': initial,
                       'dev_nll': best_nll, 'wall_s': time.monotonic() - start, 'test_dropped': dropped})
    (out / 'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    upload(out / 'release', args.hf_repo)
    upload(out, args.hf_repo, 'run')
    print(json.dumps({'status': 'quality_continuation_complete', 'repo': args.hf_repo, 'evaluation': evaluation}), flush=True)


if __name__ == '__main__':
    main()
