# Project preferences

- Prefer Google Colab G4 GPU instances for training and evaluation. Use Modal only as a fallback if Colab G4 is unavailable or unsuitable, and record why.
- Current phase: quality first, with CPU and model-size constraints out of scope. Run matched causal/joint-bidirectional Gemma continuation and a pinned larger Qwen quality reference. Defer CPU optimization and student distillation.
- CPU inference latency is a primary objective. Report measured end-to-end latency, input length, CPU model, thread count, precision, and quality together.
- Keep code, dataset manifests, and aggregate reports in GitHub. Keep credentials and private raw data out of git. Default model artifacts to private Hugging Face repositories.
- Do not describe pilot or synthetic evaluations as production financial-services validation.
