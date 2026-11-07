# Phase-3 supervised pilot

The user authorized training on November 6, 2026. This pilot trains Gemma on source training data, not on the proposed final-evaluation datasets. No new final benchmark run or Jev labeling job is launched. The already-authorized Qwen evaluation continues separately.

## Scope and labels

Seal 10,000 training decisions, 700 development decisions and 700 calibration decisions before fitting. Canonical sources are ContractNLI train/dev, RAGTruth train, TS-Bench train/validation, When2Call chosen/rejected training preferences, and existing BANKING77 training/development/calibration retention rows. Source counts, hashes, roles, exclusions, family counts and class counts are in `data/phase3-manifest.json`.

Contract hypotheses use native entailment/contradiction/not-mentioned labels. RAGTruth uses native annotated unsupported spans for response-level grounding. TS-Bench uses native `Harmfulness_Rating`, `Prompt_Injection` and `Malicious_User_Request` targets; only user request, history, current action and environment enter the model. Its answer, reward, index and outcome annotations are excluded from inputs. When2Call uses its released synthetic chosen/rejected preference supervision, not inferred free-text action labels or human gold. Randomize which string key denotes the preferred response as well as candidate insertion order.

This is an internal research mixture, not production traffic or a newly reviewed financial-service dataset. Outcome-based model-routing supervision is deferred until its RouterArena/upstream overlap and price/pool audit passes; do not claim this first pilot trains that objective. When2Call preference learning is not the native four-way action benchmark. BANKING77 retains the earlier eight-candidate adaptation, not full 77-way classification.

## Splits and exclusions

Current public-evaluation rows and historical test/holdout source contexts are exclusion-only. Documents, source prompts, tool-request families and preference conversations stay together. Preserve earlier banking development/calibration roles; a component with either role cannot become training data. Exact normalized content and approximate 64-permutation MinHash retrieval followed by exact three-word-shingle Jaccard >=0.85 define transitive source components. Components touching a known test context are excluded. This is a bounded duplicate audit, not proof against semantic or undisclosed upstream contamination.

Use research-balanced domain/class sampling without repeating selected rows. Record native cases and connected families separately from derived decisions. Reserve RouterArena, TraceSafe, the proposed LegalBench tasks, FinQA and CLINC150/OOS for later final evaluation; do not ingest their labels or templates into this training job. Future source-lineage checks are still required before making a fresh-holdout claim.

## Model and training

Start from the pinned official `google/gemma-4-E2B-it@4adf15748de9e1a0097b6274cee9c15836c41c37`, with fresh rank-16 LoRA (alpha32, dropout0.05) and a fresh shared scalar candidate readout. This does not inherit the narrow phase-2 classifier head. Fix independent causal candidate attention; this is not another mask ablation and does not establish why the earlier model failed.

Use BF16 base weights, original FP32 LoRA/readout, strict math SDPA, TF32 disabled, gradient checkpointing, one example per microbatch and effective batch eight. Train one deterministic shuffled pass, up to 1,250 optimizer updates. Initial learning rates are 1e-4 for LoRA and 3e-4 for the readout, with 50-step warmup, cosine decay and gradient clipping at one. Seed20261112. Persist the exact planned training-input-stream hash.

Support up to eleven choices and 8,192 packed tokens. Compute token eligibility before fitting, reject overlength source rows without truncation, and record exclusions. Perform a longest-eligible-example backward/memory preflight before parameter updates. Save candidate capacity/context in the artifact for exact continuation. The public API's default eight-choice guard is unchanged.

The actual pilot rows contain two, three or eight candidates; eleven is model capacity, not a validated training result. The sealed training maximum is 6,568 packed tokens. See the exact choice-count distributions in `data/phase3-preflight.json`.

## Selection, calibration and persistence

Measure development before training and every100 updates. Select lowest equal-domain mean `NLL / log(candidate count)`; this is a prespecified research objective, not inferred production weights. Report per-domain semantic macro-F1, accuracy and binary recall/false-positive rate. Fit one positive scalar temperature on the separate calibration fold only after selection. These diagnostics do not support a final superiority claim against Jev.

Checkpoint LoRA, scalar readout, optimizer, CPU/CUDA/Python RNG, step, selected checkpoint, configuration and training log every100 updates. In-place and VM recovery require identical data/configuration/code hashes and restore optimizer/RNG. Transient uploads retry before abandoning a usable GPU. Limit the owned training job to four hours and two VM recovery attempts, and automatically release completed/failed/expired GPUs.

Use an isolated Colab G4 session. Keep raw contexts in private working copies transferred directly to that VM. GitHub stores code/manifests/aggregates. The private `jsaurabh/financial-decision-phase3-pilot` dataset stores code/manifest/runtime status only; the private `jsaurabh/gemma4-finance-phase3-pilot` model stores adapters and training artifacts. Raw RAG contexts are not republished. Preserve base and source terms; this is not a production release.

## Subsequent comparison requested after training

The user subsequently asked how the reported scores compare with Jev. Run `python -m finance_decisions.phase3_jev_compare` to score Jev on the identical sealed 700-case calibration fold, without another GPU or model update. Reuse a private append-only per-case cache, validate canonical input hashes and candidate order, and send only state/question, never references, to the API. Save resolved provider versions, coverage and native decisions. Per-case outputs/probabilities stay in a separate private diagnostic directory and the private dataset's `comparisons/jev-calibration/` prefix; they are not training inputs. Public artifacts are aggregate JSON/Markdown only.

Gemma numbers come from the completed frozen training summary; no per-case Gemma outputs were retained, so do not compute paired significance or claim a fresh replay. Positive scalar temperature does not change top-1 decisions, but calibration-fit NLL would have an advantage; compare top-1 accuracy and available task error rates. The fold contains 100 decisions per task, with fewer independent source groups. Native labels, synthetic preferences and eight-candidate banking adaptations limit interpretation. The reserved final datasets remain untouched.
