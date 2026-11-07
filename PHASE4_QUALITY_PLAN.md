# Quality recovery after independent evaluation

Status: the user subsequently authorized aligned train/validation/test slices and a separately reported transfer suite. Execution now follows [PHASE4_TRAINING_PROTOCOL.md](PHASE4_TRAINING_PROTOCOL.md), which supersedes the earlier source-reservation and diagnostic-only execution scope below. CPU optimization remains deferred. Preserve the completed earlier evaluations and their full outputs, including failures.

## What the evidence says

The phase-3 pilot scored52.0% versus Jev83.3% on the equal-domain independent classification mixture. All11,634 requested classification decisions were valid for both systems; this deficit is not explained by missing predictions. Quality-focused RouterArena replay is a narrower positive result and must be preserved as a retention target. See [complete independent cohorts](results/final-eval/independent-cohorts.md).

- Training contained10,000 decisions,23 distinct instruction strings, and only2-,3- or8-candidate tasks. It contained no `choice` examples whose labels were exactly `Yes`/`No`. Different banking candidate subsets increased the number of label sets without adding new question/task semantics.
- On1,192 audit-rights and1,223 liability-cap decisions, Gemma predicted `No` every time. Other legal tasks also heavily favored `No`. This is evidence of poor transfer to those tasks/formats; it does not establish whether the dominant cause is the scalar readout, instruction/label binding, data coverage or backbone capacity.
- On955 FinQA correct/counterfactual pairs,906 pairs received the same prediction. Only26 pairs had both decisions correct, and23 had both wrong. Financial-verification average precision was0.506, close to the balanced positive prevalence0.5. Changing confidence scaling cannot create the missing ranking signal.
- CLINC OOS recall was22.0% versus Jev95.7%. Gemma's67.8% accuracy was below the81.8% always-in-scope baseline. Both coverage and class-sensitive baselines must be routine selection checks.
- The earlier Qwen27B reference was substantially stronger than the earlier Gemma variants on several development-informed public tasks, sometimes exceeding Jev. It has prior upstream benchmark exposure, so those results justify a quality baseline, not a clean superiority claim or a guaranteed winner on fresh tests.

The evidence supports weak transfer and substantial task-coverage gaps. It does not yet prove that a larger backbone, a different attention mask or more training steps alone will solve them.

## Ordered work

### 1. Establish a trustworthy diagnostic baseline

Reproduce the frozen artifact's original calibration metrics with the current inference path, documenting numerical tolerance. Compare the scalar-candidate model with the same instruction base in ordinary generative classification mode on a NEW, training-disjoint diagnostic development set. Add the existing larger Qwen reference as another quality baseline. These are diagnostics; do not change or replace existing final receipts.

Use mechanically labeled examples to test whether predictions follow the state/question and candidate meanings: positive/negative pairs, negated predicates, `Yes`/`No` versus `true`/`false`, arbitrary candidate IDs with complete descriptions, candidate reordering, equivalent JSON/text representations and unseen instruction wording. For numerical verification, the proposed value must change while evidence stays fixed; the expected decision must flip. Record paired semantic consistency as well as accuracy. Use these controls to distinguish an inference/readout failure from a learning/capacity failure before scaling training.

### 2. Build training data for the actual decision capabilities

Use separate, authorized training sources and new document/request families. Keep TraceSafe, RouterArena and reserved final benchmark sources evaluation-only. Audit provenance, licensing, exact/near overlap and long-text containment before ingestion. A native training split from a reserved evaluation-only source is not automatically authorized for ingestion by this plan.

Cover legal clause/policy decisions, financial numeric verification, intent/OOS rejection and tool-trajectory authorization/validity. Include genuine benign controls, near-boundary negatives, insufficient-evidence cases and ambiguity. Model-routing training labels must come from measured downstream model outcomes; prompt complexity and a teacher's intuition are not oracle routing labels. Keep current routing performance as a retention target.

Expand instruction and answer-schema diversity within every capability. Balance semantic positive/negative conclusions and candidate aliases, randomize candidate order with correctly remapped labels, and vary candidate counts. Strip reward metadata, mutation annotations, source-model hints and other gold-adjacent fields. Keep paraphrases, counterfactuals and all document/trajectory variants in one source family before splitting.

Use native annotations and executable arithmetic/program checks where possible. A stronger teacher can supply additional labels, concise evidence-based explanations and hard negatives on TRAINING inputs. Independently check those labels; Jev is a comparator, not ground truth. Legal/policy ambiguity and consequential tool authorization require review rather than trusting synthetic labels by default. Do not use final-case answers, Jev final predictions or mutated final inputs for distillation.

### 3. Compare learning approaches under controlled development selection

Start with a bounded diverse-data pilot and a measured learning curve. Select data volume and training duration using development improvement, not an assumed example-count requirement. Compare the current Gemma backbone/readout with a stronger Qwen candidate on matched training/development data and a documented compute budget. Measure the ordinary generative baseline too; retaining generative reasoning or adding an evidence/reasoning objective may help if decision-only supervision loses necessary capabilities.

Do not make mask changes, model size, curriculum and loss all change in one unexplained comparison. Prioritize the smallest controlled changes that distinguish the failure hypotheses. Treat a larger model as the quality reference while CPU/size constraints are deferred. Keep Gemma as a candidate/student if it can retain the necessary quality; do not force a deployment-size decision before the quality result exists.

### 4. Select for transfer and business-relevant errors

Create separate training, development and calibration source families. Include source-held-out and instruction/label-schema-held-out development tests, not only new rows in the same23 templates. Development gates must exceed constant baselines, improve correct/counterfactual discrimination, reduce OOS misses, control benign false blocks and retain routing quality. Report per-task/class results, source-family uncertainty and coverage rather than selecting solely on a pooled accuracy.

Use a documented selection rule for checkpoint/curriculum/model choice. Calibrate confidence and business-specific operating thresholds on a separate calibration set only after the underlying classifier separates classes. Do not select thresholds using final benchmark labels.

### 5. Earn a fresh superiority claim

The current benchmark results now inform development and must be described accordingly in subsequent iterations. Repeated gains on them are useful regression evidence, not a new untouched confirmation. Before the next claimed winner, seal a fresh representative holdout with disjoint sources/documents/requests, known sampling, native or independently reviewed references, and appropriate financial-workflow error costs. Private consequential-workflow claims require adjudicated representative cases; synthetic public mutations do not substitute for them.

Freeze the winning checkpoint, preprocessing, prompts, candidate rules, confidence calibration and coverage policy before new final predictions. Compare against the same resolved Jev provider on identical payloads. Require positive paired source-family evidence on the workflows claimed, adequate precision, no hidden coverage reduction, and acceptable benign false blocks. Routing must use realized downstream quality/cost with its own uncertainty. Report losses and inconclusive tasks explicitly; a favorable average cannot establish superiority on a losing workflow.

## Immediate next deliverable

A reproducible diagnostic package with calibration replay, known-truth semantic/format controls, matched Gemma generative and larger-Qwen baselines, and an audited new training/development manifest. No new large training run is launched by this design document. Existing final and TraceSafe jobs retain their bounded GPU lifecycles and private per-case artifacts.
