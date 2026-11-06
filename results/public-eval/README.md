# Public evaluation: background run

Evaluation-only run over 26,196 frozen decisions. One Colab G4 worker runs the two Gemma checkpoints; a separate G4 runs Qwen 27B; Jev runs through the API in parallel.

[Interim results](PROGRESS.md) contain the integrity-checked completed causal-Gemma/Jev comparison. Joint Gemma and Qwen are not complete in that snapshot.

- [Frozen protocol](../../PUBLIC_EVAL_PROTOCOL.md)
- [Corpus manifest](../../data/public-eval-manifest.json)
- [Preflight token lengths](../../data/public-eval-lengths.json)
- [Private live checkpoints](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval/tree/main/runs)

Aggregate results will be written here as `REPORT.md`, `summary.json` and pinned `artifacts.json`. Per-case predictions stay private. No raw benchmark text is republished. The background report worker checks matching inputs/candidate order and commits aggregate results to the project branch after the run.

Workers save progress every 500 decisions. An owned-session controller sends kernel heartbeats, allows up to two recoveries for Gemma and three for Qwen, and stops completed or terminally failed GPUs. Qwen's third recovery addresses a checkpoint DNS failure and a heartbeat error previously misclassified as VM loss. Successful status downloads now override heartbeat failures; three consecutive status-download failures are required for recovery. New workers can resume locally after evaluator exits without replacing the GPU. An eight-hour GPU deadline bounds resource use. The report collector retries transient polling failures and publishes a partial report if a worker exhausts recovery.

A changing Colab image caused an initial Gemma torch/torchvision import failure; that worker was stopped and a new G4 was started with matching pinned versions. The first Qwen session was subsequently lost after saving 2,124 records; the replacement G4 resumes those private checkpoints. These interruptions do not change the frozen evaluator, inputs or model revisions.

Local background control files are under `/workspace/public-eval-control/`: `status.json` tracks the GPU lifecycle, `report-status.json` tracks collection, and the detached logs are under `/workspace/public-eval-source/`. The controller only manages the two sessions named `public-eval-gemma` and `public-eval-qwen`.

The report collector initially copied checkpoints into the live local Jev output directory, which could overwrite newer predictions. It now downloads into a separate private cache. Recovery retained 23,239 verified records and re-requests only the 2,957 missing cases, without selecting by accuracy. Completion checks require every expected case ID exactly once.
