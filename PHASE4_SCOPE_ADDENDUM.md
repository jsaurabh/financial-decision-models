# Phase4 candidate scope update

On 2026-11-06 (America/Los_Angeles), after launch and before either phase4 candidate had been evaluated on the frozen core/transfer tests, the user clarified that the current training focus is Gemma. The earlier request for simultaneous Gemma/Qwen work is not carried forward to this round.

The Qwen G4 runtime and its lifecycle controller were stopped. Existing logs and any already uploaded artifacts are retained. Qwen is excluded from the active comparison; it is not scored as a model with missing predictions. No replacement Qwen training run is scheduled.

The Gemma worker continues with the original sealed data, initialization, training recipe, validation checkpoint selection, calibration and final-test protocol. No dataset, training parameter or claim threshold changes. Gemma is the sole candidate compared with Jev; validation selects its checkpoint. The separately reported transfer suite remains evaluation-only.

This addendum changes candidate scope only. The original `PHASE4_TRAINING_PROTOCOL.md` and manifest remain byte-for-byte intact to preserve their frozen hashes and Gemma resume compatibility. Their references to a two-candidate comparison and selecting a winning backbone are superseded by this addendum.
