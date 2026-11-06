# Financial-services decision models

Gemma 4 E2B adapted into a text-only, schema-driven classifier: one forward pass returns a probability distribution over supplied choices. It never generates an answer string. This independently follows the decision-model pattern; it does not reproduce TypeSafe Jev's undisclosed architecture or weights.

**Current priority: quality first.** Training and evaluation prefer **Colab G4**. CPU quantization, distillation, and latency optimization are deferred until the quality baseline is reviewed. Preferences are saved in [AGENTS.md](AGENTS.md).

**Current release: [Gemma 4 E2B financial decisions v4](https://huggingface.co/jsaurabh/gemma4-e2b-finance-decisions-v4)** (private weights). On a fresh 1,419-example public banking/legal holdout it scored **94.15%**, compared with **93.94% for live Jev 1.14.0**. The difference is three examples; this is roughly tied on this benchmark. Jev remains stronger on the separate policy/arithmetic and routing probes. Read the [quality report](results/QUALITY.md) for per-domain results and limitations.

## Architecture and scope

- Official `google/gemma-4-E2B-it` backbone at a pinned revision; text tower only at inference.
- Rank-16 LoRA on attention and MLP projections, plus a shared scalar readout over each candidate's final token in v4.
- `choice`, binary `noul`, and ordered `score` requests; 2–8 supplied candidates per question. Invalid candidate positions are masked.
- Candidate order is randomized during training and checked during evaluation. The v4 scorer achieved 99.93% semantic prediction agreement under candidate reordering on the fresh public holdout; small numerical differences can affect ties.
- Temperature is fitted on a separate calibration split. Probabilities are conditional on the supplied schema. They are not guaranteed probabilities of correctness on unseen tasks.

The v4 candidate head replaces the earlier position-specific eight-way matrix with a **shared scalar scorer**. Candidates share the state/question prefix, have separate causal attention blocks, and reset their rotary positions at the prefix boundary. One candidate cannot attend to another. This preserves a single forward pass while making each candidate's computation independent of its list position. Native Gemma tests verify permutation invariance, padding invariance, and gradients. Earlier positional-readout runs are retained as baselines.

This scorer judges each description against the supplied state and instructions. Questions that require one candidate to inspect another candidate's text need a different attention layout.

The initial mixture covers banking support intents, legal clauses and terms, and hypothetical policies for AML controls, payments, credit operations, insurance, investment operations, privacy, financial reporting, and legal workflow. Model routing uses supplied capability descriptions rather than a measured leaderboard of commercial models.

## Data and evaluation

[data/manifest.json](data/manifest.json) pins public source revisions, split hashes, and counts. Raw training data are excluded from git. The initial mix contains 13,513 train, 1,472 calibration, and 1,712 test rows before length filtering. Development families are removed from the training split before optimization.

Public BANKING77 and LEDGAR tasks use **eight sampled candidates including the reference**, not full 77/100-class benchmarks. Synthetic cases use explicit, rule-derived labels; they are not human reviewed or production financial records. Aggregate accuracy can be dominated by easier synthetic tasks, so read the per-domain results.

The quality continuation adds only the **training** families from the project's earlier contrastive banking curriculum. A separate frozen 128-case banking set verifies exact input hashes against archived Jev and 27B-model predictions. Those provider results are historical comparisons, not fresh API calls. The frozen benchmark supplies no training or calibration labels.

Live Jev 1.14.0 predictions are also collected on the same inputs through the user's API key. [results/jev-live/summary.json](results/jev-live/summary.json) records successful API calls and resolved model versions. [results/gemma-vs-jev.json](results/gemma-vs-jev.json) checks exact input hashes, candidate order, and reference labels before comparison. It deduplicates repeated semantic inputs; the raw benchmark reports are retained separately.

The later [public holdout](data/public-fresh-manifest.json) adds 1,419 previously uninspected examples covering 71 banking intents and 100 contract topics, plus legal-term predicates. It excludes all previously used rows and normalized states. These references are not read by the candidate-head trainer or used to select checkpoints. The original pilot's first 240 BANKING77 test rows covered only six intents; the new stratified holdout broadens that coverage.

The original pilot test has informed the development plan, so later improvements on it are development results. The synthetic robustness probe was first evaluated after V3 and informed the V4 architecture decision, so it is now a development probe. Its labels remain excluded from training, calibration, and checkpoint selection. The separate public-fresh holdout is reserved for the final V4 evaluation.

Run the corpus audit after preparing private files:

```bash
PYTHONPATH=. python scripts/audit_data.py
pytest -q tests
```

## Reproduce on Colab G4

Use a G4 runtime and Python with CUDA-enabled PyTorch. The completed runtime package versions and GPU identity are saved in each run configuration. Colab's preinstalled `torchao` can conflict with current PEFT; it is unused in this recipe and may be removed if the PEFT preflight reports incompatibility.

```bash
pip install -r requirements.txt
python -m finance_decisions.build_data
python -m finance_decisions.train --steps 700
```

Set `HF_TOKEN` through Colab Secrets or your environment, never in notebook source. Artifacts default to private repositories in the authenticated owner's namespace. The trainer refuses to upload to a public model repository.

The automated session scripts stage a source archive and a separately transferred credential file, delete that credential file after loading it, and start a bounded background process. `scripts/colab_job.py` expects `/content/finance-source.tar.gz` and `/content/.finance-hf-token`. Use `scripts/colab_status.py` or download the log through the Colab CLI to inspect progress. Release the G4 runtime after verifying the final artifacts.

The quality continuation additionally expects normalized `data/private/contrastive-banking-train.jsonl` from the upstream MIT-licensed training corpus. [data/contrastive-banking-manifest.json](data/contrastive-banking-manifest.json) records its hash. Prepare the earlier frozen test with `scripts/prepare_legacy_eval.py` when the companion `qwen-rlcd-jev-repro` checkout is available.

```bash
python -m finance_decisions.harden_quality
python -m finance_decisions.evaluate_quality \
  --artifact runs/gemma4-finance-v2/release \
  --out results/gemma4-finance-v2 \
  --hf-repo jsaurabh/gemma4-e2b-finance-decisions-v2
```

For paraphrase hardening, continue the selected adapter and fit calibration again after merging the release weights:

```bash
python -m finance_decisions.harden_quality \
  --starting-artifact runs/gemma4-finance-v2/selected \
  --out runs/gemma4-finance-v3 \
  --hf-repo jsaurabh/gemma4-e2b-finance-decisions-v3 \
  --augment --steps 700
```

The shared candidate-head experiment starts from that adapter with `--candidate-head` and saves to a separate versioned repository. Its packed token budget is 1,536; the model still returns at most eight candidate probabilities.

```bash
python -m finance_decisions.build_public_holdout
python -m finance_decisions.harden_quality \
  --starting-artifact runs/gemma4-finance-v3/selected \
  --out runs/gemma4-finance-v4 \
  --hf-repo jsaurabh/gemma4-e2b-finance-decisions-v4 \
  --augment --candidate-head --steps 700
python -m finance_decisions.evaluate_quality \
  --artifact runs/gemma4-finance-v4/release \
  --out results/gemma4-finance-v4 \
  --hf-repo jsaurabh/gemma4-e2b-finance-decisions-v4 \
  --fresh-probe --public-holdout
```

The complete live comparison can be reproduced without sending reference labels to the API:

```bash
python -m finance_decisions.robustness
# TYPESAFE_API_KEY must be set in the environment.
python -m finance_decisions.evaluate_jev
python -m finance_decisions.compare_quality
```

Jev's native returned choice determines top-1 accuracy. Probability metrics use its returned probabilities, clipped at `1e-12` for log loss. Its provider-specific confidence value is recorded separately. API wall times include the network and are not compared with local CPU or GPU latency.

## Call the classifier

The local API loads a saved merged artifact. GPU inference is the current validation target; CPU performance has not been established.

```bash
export DECISION_ARTIFACT=/path/to/saved/release
uvicorn finance_decisions.serve:app --host 127.0.0.1 --port 8000 --workers 1
```

```json
{
  "state": "The customer cannot find a payment they authorized yesterday.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the best support route.",
      "criteria": {
        "payment_operations": "Investigate a missing or pending payment.",
        "card_security": "Investigate an unrecognized card transaction.",
        "human_review": "Use when evidence is insufficient or conflicts."
      }
    }
  }
}
```

POST to `/v1/decisions` (`/v1/systemone` is an alias). The response includes the selected value, all candidate probabilities, and a configurable `needs_review` flag. Its default threshold is an experimental setting; select production thresholds on reviewed in-domain examples. Multi-question calls currently run each question separately. Long requests are rejected explicitly instead of silently truncated.

## Interpretation

This is a research-quality baseline, not a validated underwriting, sanctions, investment, or legal decision system. New question schemas and financial workflows need their own labeled evaluations. No private production data were supplied. Public-data pretraining contamination and more subtle semantic overlap cannot be ruled out by the exact-match split audit.

The [Airbnb benchmark](https://airbnb.tech/ai-ml/frontier-quality-at-1-of-the-cost-a-production-benchmark-of-llm-classifiers/) motivates domain fine-tuning and probability-based evaluation. Its quality results use a much larger production dataset, and its latency results use a GPU; neither transfers automatically to this pilot.

Attribution and source terms are in [THIRD_PARTY.md](THIRD_PARTY.md).
