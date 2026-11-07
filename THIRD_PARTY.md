# Sources and attribution

- Gemma 4 E2B: Google DeepMind, `google/gemma-4-E2B-it`, Apache 2.0. The exact base revision is recorded in `finance_decisions/model.py`.
- MiniLM L6 H384 uncased: Microsoft. The downloaded revision is recorded in each student evaluation. The student is an independent distillation, not Jev weights or architecture.
- BANKING77: PolyAI, Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders,” 2020, CC-BY-4.0. The pilot samples eight candidates including the reference from 77 intents.
- LexGLUE: Chalkidis et al., “LexGLUE: A Benchmark Dataset for Legal Language Understanding in English,” 2022, `coastalcph/lex_glue`, CC-BY-4.0 dataset card. Underlying LEDGAR and UNFAIR-ToS sources retain their original terms. Raw corpora are not redistributed in GitHub.
- llama.cpp: ggml-org, MIT. Optional conversion is pinned separately. No upstream source is vendored here.

Synthetic policy and capability-routing cases are authored for this project. They describe hypothetical supplied policies, not legal requirements or model-provider rankings. No customer records are included.

Design reference: Airbnb Engineering, [Frontier quality at 1% of the cost: a production benchmark of LLM classifiers](https://airbnb.tech/ai-ml/frontier-quality-at-1-of-the-cost-a-production-benchmark-of-llm-classifiers/). Its latency measurements use a GPU and are not a CPU performance guarantee.
