# Frozen Qwen reference loader

`autojev/model.py`, `autojev/types.py`, and `autojev/__init__.py` are copied without modification from `jsaurabh/qwen-rlcd-jev-repro` commit `daa6e77` (`colab/noncausal_27b/autojev/`). MIT license retained. The small loader adapts that repository's evaluation loader to use the model module's built-in saved attention mode and pooling rather than importing a training script. It does not modify weights, tokenization, masks or scoring.

The released adapter is pinned in `finance_decisions.phase2_eval`. Use a separate environment with the reference requirements; Gemma training retains its original runtime. Reference requirements are pinned from the source project; add datasets/scipy/scikit-learn/requests for the shared evaluator. This is a differently trained, larger model, not a matched causal comparison.
