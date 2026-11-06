# Public evaluation: background run

Evaluation-only run over 26,196 frozen decisions. One Colab G4 worker runs the two Gemma checkpoints; a separate G4 runs Qwen 27B; Jev runs through the API in parallel.

- [Frozen protocol](../../PUBLIC_EVAL_PROTOCOL.md)
- [Corpus manifest](../../data/public-eval-manifest.json)
- [Preflight token lengths](../../data/public-eval-lengths.json)
- [Private live checkpoints](https://huggingface.co/datasets/jsaurabh/financial-decision-public-eval/tree/main/runs)

Aggregate results will be written here as `REPORT.md`, `summary.json` and pinned `artifacts.json`. Per-case predictions stay private. No raw benchmark text is republished. The background report worker checks matching inputs/candidate order and commits aggregate results to the project branch after the run.

Workers save progress every 500 decisions and are watched by an owned-session lifecycle controller that stops completed or failed workers. A changing Colab image caused an initial Gemma torch/torchvision import failure; that worker was stopped and a new G4 was started with matching pinned versions. Qwen and Jev continued independently.
