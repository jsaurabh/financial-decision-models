"""Bounded LoRA pilot, held-out calibration, order controls, and teacher export."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model

from .metrics import fit_temperature, metrics
from .model import DecisionModel, counts, targets
from .schema import render


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def shuffled_options(rows, rng):
    rows = copy.deepcopy(rows)
    for row in rows:
        if row['question']['type'] == 'choice':
            pairs = list(row['question']['criteria'].items()); rng.shuffle(pairs)
            row['question']['criteria'] = dict(pairs)
    return rows


def filter_rows(model, rows):
    accepted, dropped = [], Counter()
    for row in rows:
        # Leave a small margin for a shuffled rendering's token-boundary changes.
        n = model.token_length(row)
        if n <= model.max_length - 8:
            accepted.append(row)
        else:
            dropped[row['domain']] += 1
    return accepted, dict(dropped)


@torch.inference_mode()
def infer(model, rows, batch_size=16):
    model.eval(); result = []
    for start in range(0, len(rows), batch_size):
        part = rows[start:start+batch_size]
        result.extend(model(model.prepare(part), counts(part)).float().cpu().tolist())
    return result


def report(model, rows, logits):
    result = metrics(logits, targets(rows), model.temperature)
    result['domains'] = {}
    for domain in sorted({r['domain'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['domain'] == domain]
        result['domains'][domain] = metrics([logits[i] for i in indices], targets([rows[i] for i in indices]), model.temperature)
    return result


def upload(path, repo, prefix=''):
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise ValueError('Refusing to upload pilot weights to a public repository')
    api.upload_folder(repo_id=repo, folder_path=str(path), path_in_repo=prefix,
                      commit_message='Save financial decision pilot artifact')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--micro-batch', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--out', default='runs/gemma4-finance-v1')
    parser.add_argument('--hf-repo', default='jsaurabh/gemma4-e2b-finance-decisions-v1')
    args = parser.parse_args()
    random.seed(20261105); np.random.seed(20261105); torch.manual_seed(20261105)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = DecisionModel.pretrained_gemma().to('cuda')
    model.backbone.requires_grad_(False)
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none'))
    splits, dropped = {}, {}
    for split in ['train', 'calibration', 'test']:
        splits[split], dropped[split] = filter_rows(model, read(f'data/private/{split}.jsonl'))
    train = splits['train']
    # Dedicated development families; calibration never selects a checkpoint.
    dev_families = {r['family'] for r in train if int(hashlib.sha256(r['family'].encode()).hexdigest()[:8], 16) % 20 == 0}
    dev = [r for r in train if r['family'] in dev_families]
    train = [r for r in train if r['family'] not in dev_families]
    rng = random.Random(20261105); rng.shuffle(train)
    cfg = {'steps': args.steps, 'effective_batch': args.micro_batch * args.grad_accum,
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'train_rows': len(train), 'dev_rows': len(dev), 'dropped_overlength': dropped,
           'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
           'data_manifest': json.loads(Path('data/manifest.json').read_text())}
    (out / 'run_config.json').write_text(json.dumps(cfg, indent=2)); print(json.dumps(cfg), flush=True)
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 1e-4},
        {'params': model.readout.parameters(), 'lr': 5e-4}], weight_decay=.01)
    start_time = time.monotonic(); best_nll = float('inf'); best_step = 0
    logs = []
    for step in range(1, args.steps + 1):
        model.train(); opt.zero_grad(set_to_none=True); losses = []
        warm = min(1., step / 35)
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps))
        opt.param_groups[0]['lr'] = 1e-4 * warm * decay
        opt.param_groups[1]['lr'] = 5e-4 * warm * decay
        for micro in range(args.grad_accum):
            first = ((step - 1) * args.grad_accum + micro) * args.micro_batch
            rows = shuffled_options([train[(first + j) % len(train)] for j in range(args.micro_batch)], rng)
            logits = model(model.prepare(rows), counts(rows))
            loss = F.cross_entropy(logits, torch.tensor(targets(rows), device='cuda'))
            (loss / args.grad_accum).backward(); losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        entry = {'step': step, 'loss': sum(losses) / len(losses), 'wall_s': round(time.monotonic() - start_time, 2)}
        logs.append(entry)
        if step % 10 == 0 or step == 1:
            print(json.dumps(entry), flush=True)
        if step % 175 == 0 or step == args.steps:
            dev_logits = infer(model, dev)
            score = metrics(dev_logits, targets(dev))
            print(json.dumps({'step': step, 'dev': score}), flush=True)
            if score['nll'] < best_nll:
                best_nll, best_step = score['nll'], step
                model.save(out / 'selected')
            model.save(out / 'checkpoint')
            torch.save({'step': step, 'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': rng.getstate()}, out / 'checkpoint' / 'training_state.pt')
            (out / 'training.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in logs))
            upload(out / 'checkpoint', args.hf_repo, 'checkpoint')
    del model, opt; torch.cuda.empty_cache()
    model = DecisionModel.load(out / 'selected', device='cuda')
    cal_logits = infer(model, splits['calibration'])
    model.temperature = fit_temperature(cal_logits, targets(splits['calibration']))
    test_logits = infer(model, splits['test'])
    evaluation = report(model, splits['test'], test_logits)
    reordered = shuffled_options(splits['test'], random.Random(42))
    reordered_logits = infer(model, reordered)
    evaluation['option_order_control'] = report(model, reordered, reordered_logits)
    evaluation['temperature'] = model.temperature
    evaluation['selected_step'] = best_step
    evaluation['wall_s'] = time.monotonic() - start_time
    evaluation['peak_cuda_gib'] = torch.cuda.max_memory_allocated() / 2**30
    (out / 'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    model.save(out / 'release', merge=True)
    (out / 'release' / 'README.md').write_text('---\nlicense: apache-2.0\nbase_model: google/gemma-4-E2B-it\npipeline_tag: text-classification\n---\n# Financial decisions: Gemma 4 E2B pilot\nPrivate experimental classifier. Load with finance_decisions.model.DecisionModel. Text-only, at most 8 candidates and 384 tokens. No production financial validation.\n')
    upload(out / 'release', args.hf_repo)
    # Aggregate results and configs stay private on HF; only reports go to git.
    upload(out, args.hf_repo, 'run',)
    print(json.dumps({'status': 'complete', 'repo': args.hf_repo, 'evaluation': evaluation}), flush=True)


if __name__ == '__main__':
    main()
