# Phase 3: decision quality on financial-service workflows

This is the next-round plan, not a report of a new trained model. Finish the running public evaluation. Do not change its frozen model, temperature, prompts or inputs. CPU optimization remains deferred; prefer Colab G4. Dataset preparation and diagnostics can proceed on CPU in parallel.

## What the current evidence says

The phase-2 continuation used 6,131 distinct training decisions across banking intents, legal topics/predicates, declared-capability routing and generated financial rules. It consumed 2.64 million packed tokens. That mixture is substantially different from tool-action safety, evidence-grounded contract NLI, hallucination detection and outcome-based model routing. The current test scores do not isolate data, architecture, initialization or capacity as the cause.

The completed causal Gemma run has severe class bias:

- R-Judge: detected 1 of 301 unsafe interactions; predicted safe on 543 of 571 inputs.
- RAGTruth: detected 4 of 943 unsupported responses; predicted supported on 2,670 of 2,700 inputs.
- ContractNLI: predicted entailment on 1,607 of 2,091 hypotheses; not-mentioned recall is 15.5%.
- On the complete 571-case R-Judge set, our larger Qwen reference scores 85.8%, Jev 87.2%, and causal Gemma 42.7% on identical inputs. This supports testing a stronger starting decision model. It does not prove a size effect: prior data, head and architecture differ.

See [verified interim metrics](results/public-eval/PROGRESS.md) and [class/paired diagnostics](results/public-eval/gap-audit.json). Remaining Qwen comparisons use incomplete length-selected subsets; do not treat those as full-dataset scores. The public evaluation has now informed development and is an external regression suite, not a fresh blind test for future iterations.

## Define the target by workflow

| Track | Intended decisions | Primary quality comparison against Jev |
|---|---|---|
| Banking support and tool safety | Intent, tool/action choice, argument completeness, authorization, escalation and prompt-injection risk | Unsafe-action recall at a fixed false-block rate; intent macro-F1; out-of-scope and missing-information handling |
| Legal and financial policy verification | Entailed/contradicted/not mentioned, exceptions, evidence sufficiency, grounded claims | Per-class recall/F1, document-grouped evaluation, contradiction and missing-evidence errors, calibration |
| Model routing | Cheapest adequate model among the actual eligible models | Realized downstream quality at fixed cost, cost at fixed quality, oracle regret and routing failures |

Set numerical operating constraints with representative labeled workflows, not arbitrary benchmark averages. Banking intent, financial advice, contract-topic labeling and contract entailment are different tasks. Public benchmarks can support development but cannot replace a reviewed financial-service test set. Do not invent production traffic weights or business error costs.

## 1. Rule out a representation or implementation fault

Use a separate diagnostic/development set with simple unambiguous controls, never a newly claimed blind test derived from observed failures. Check binary polarity, label remapping and class balances, the trained readout, option reordering, and short versus long context. Compare the current readout to the instruction base model's answer likelihood on the same development examples. Correct implementation faults before interpreting a training run as a capacity result.

R-Judge inputs all fit the old 2,048-token envelope and most RAGTruth inputs do too, so increasing context alone cannot explain or repair their gap. The public run extends candidates/context without training; future data must exercise the intended candidate counts and full context lengths. Keep strict math SDPA consistent through training, selection, calibration and evaluation. Existing native-backend temperatures are not evidence of new-domain calibration.

## 2. Build a broad decision foundation and a focused domain curriculum

Start with source-authorized training partitions: ContractNLI train/development, canonical RAGTruth training responses, ToolSafe/TS-Bench training, When2Call SFT/preference releases, and source-label agent-tool decisions. Add broad NLI, grounding and instruction-conditioned classification only after a license/source/split audit. The three suggested MaziyarPanahi releases are useful candidates, but their titles do not establish task relevance; use the upstream labels and provenance. Teacher columns must never become model inputs or independent gold labels.

For financial specialization, add reviewed policy/contract decisions, banking tool calls and grounded counterfactuals. Pair nearby examples that change the correct answer: authorized versus unauthorized recipient, quoted instruction versus user authorization, missing versus verified evidence, applicable exception versus ordinary rule, supported claim versus plausible unsupported claim. Include safe examples so high recall does not come from blocking everything. Use deterministic oracles for arithmetic and explicit policy calculations where possible; preserve whether a result is model-only or a workflow with deterministic features/tools.

Use research-balanced task/class sampling and varied templates, labels, candidate order, option counts and evidence lengths. Preserve provenance and sampling weights; do not imply the training mixture reflects production prevalence. Start with an audited 10k pilot, then 50k/100k learning-curve points if the pilot improves independent development quality. These are proposed budgets, not already available clean unique counts. Avoid multiple epochs of near-duplicate template expansions as a substitute for independent cases.

Split by original document, policy family, conversation, agent task and routing request across *all* releases and transformations. Exclude current evaluation source families and all historical holdouts. Audit exact and near duplicates; keep counterfactual families together. A source inventory alone is not training approval or proof of no leakage.

## 3. Establish the quality model, then the compact model

Train a bounded supervised pilot from the strongest suitable 27B decision checkpoint/reference after its evaluation finishes, alongside a Gemma pilot on the same logical decisions. Preserve each architecture's native representation and report common eligibility, actual tokens and training budgets; do not describe this as a clean size ablation. Compare a fresh official-Gemma/readout initialization with narrow-checkpoint continuation on development before choosing the Gemma training path if the diagnostic supports that comparison.

Use hard source labels as the primary objective. A stronger teacher can provide complete candidate distributions and explanations for independently reviewed training cases, subject to applicable source/provider terms. Use temperature-aware KL plus hard-label loss only after teacher quality is measured on the target tasks. Teacher disagreements identify annotation candidates; they are not proof the reference is wrong. A student trained only to imitate Jev has no reason to systematically correct Jev's mistakes. To exceed Jev, include independent correct supervision, financial policy/evidence and independently verified cases that Jev misses.

Distill the winning quality model into Gemma once the teacher meets task-specific gates. Since CPU size is currently out of scope, retain a larger model if it performs better. Keep the attention formulation fixed during the data experiment; the earlier mask comparison was inconclusive. Defer RL until there is a validated reward/objective and a clear SFT plateau.

## 4. Make the success claim reviewable

Freeze source-disjoint development, calibration and new final-test groups before training. Select model and curriculum on development. Fit temperatures and operating thresholds on calibration only. Use untouched documents, conversations/policy variants and ideally reviewed financial-service examples for final confirmation. Score Jev on the identical evidence, question, candidates and coverage. Record provider version. If Jev helped label training examples, keep evaluation labels independently adjudicated.

Report each workflow, class and operating point, including false negatives, false blocks, calibration, abstention/coverage, full-context failures and paired family-level uncertainty. For routing, freeze the eligible model pool, downstream outcomes and price/quality objective and include classifier charges. Do not substitute agreement with Jev, a pooled accuracy, different fallback coverage or a test-tuned threshold for superiority.

The release target is to match or exceed Jev on prespecified workflow metrics with uncertainty narrow enough to support that claim, while preserving all safety and coverage gates. A high-confidence wrong prediction on a new task must not bypass review. CPU compression and speed work follow a quality-qualified checkpoint, with the same gates rerun after compression.

## Work started

The complete causal/Jev outputs have been validated and the failure-class audit saved. A CPU-only preliminary [source inventory](data/phase3-source-audit.json) has completed without model API calls. It finds 15,090 canonical RAGTruth training responses across 2,514 normalized source contexts, 15,000 When2Call SFT rows and 9,000 preference rows (not independent counts), and 2,194 TS-Bench training rows plus 789 validation rows in a format still requiring a target/parser audit. ContractNLI has 423 training documents/7,191 decisions, including one document matching the test set exactly; exclude that document and group duplicate documents before fitting. These raw counts are not a clean eligible-corpus claim. No new weights, training run, teacher-label purchase or test-selected threshold is part of this planning update.
