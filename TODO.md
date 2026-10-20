# Research roadmap and large experiment queue

This is the forward-looking work plan for `xai-for-integrity`. It is written for the next researcher who will pull the repository and implement, launch, and analyze the planned work. The repository contains some ready analysis scripts and a historical exploratory run, but the large runs below are **planned**; they have not been launched and must not be described as results.

## Project question and two method lines

**Question.** Can visual explanations for integrity auditing be made both spatially grounded and causally tied to the evidence they cite, and does that help an independent auditor catch unsupported claims?

Keep two method lines distinct until the final comparison:

1. **G/P: supervised spatial grounding.** G adds region supervision to the existing LEGION objective. P is the planned gradient-routing control that prevents the auxiliary grounding loss from changing the native mask branch. This line tests whether evidence can be localized reliably.
2. **SAE-Causal: sparse causal evidence features.** Extract features from the exact vision encoder used by the model, identify features that causally affect a claim token, and map those features back to image patches. First evaluate this post hoc. Add a claim-to-feature selector/gate only if the post-hoc feature and causal-specificity checks pass. SAE is a candidate implementation, not a requirement: if it fails the gates, compare a simpler sparse probe/dictionary or another interpretable feature method before investing in the gated variant.

The intended final study compares native baseline, G, SAE-Causal, and the combination G+SAE-Causal. Localization, intervention faithfulness, text quality, calibration, and auditor utility are separate outcomes. A better mask score alone does not establish faithful explanations or improved integrity auditing.

## Status legend

- **Ready analysis:** a script exists and can be run when its input files are supplied.
- **Implementation required:** code/configuration and its review gate are still missing.
- **Blocked by data/protocol:** choose, license, freeze, and document the dataset or labels before the experiment.
- **Planned run:** no result exists yet. A row becomes complete only after logs, raw predictions, analysis outputs, and limitations are recorded.

The current code has a seed-42 launcher and evaluation path. It does not yet provide a reviewed multi-seed job orchestrator, a portable dataset manifest, or the P/SAE methods. See `docs/REPRODUCE.md` before asking another lab to run jobs. The current 1,000-image SynthScars test subset has already been used during method iteration; it is exploratory, not a confirmatory test set.

## Run order and gates

### Stage 0 — Make the experiment portable and freeze the protocol

- [ ] Replace lab-specific path assumptions with a validated manifest/config interface, while retaining pinned code/model hashes. Verify a clean checkout can locate data and weights without editing source files.
- [ ] Record the split by source/scar family, file IDs, source IDs, exclusions, and checksums. Check train/validation/test and external-domain source disjointness.
- [ ] Freeze the model revision, preprocessing, prompt/template, update count, optimizer, batch policy, checkpoint rule, thresholds, metric definitions, and analysis commit in a versioned protocol file.
- [ ] Implement config validation and run-ID output directories. Every run records git SHA, config, seed, data-manifest hash, hardware, start/end time, exit status, and output checksums.
- [ ] Run a one-batch data/model smoke check and a short engineering preflight. Confirm peak memory, throughput, checkpoint/resume behavior, and output schema. This is not evidence for any scientific claim.
- [ ] Do not start a full run unless the data, portability, and P-gradient gates relevant to that run pass. Keep failures and partial runs; do not silently retry or drop them.

**Deliverable:** a frozen protocol/manifest, a clean-checkout reproduction record, and measured job cost. Only after this stage can GPU-hour and storage estimates be stated from evidence.

### Stage 1 — Large confirmatory training comparison (E1)

**Purpose:** test whether spatial grounding is reproducible across training seeds, and whether P protects the native mask branch.

| Arm | Definition | Seeds | Full training jobs |
|---|---|---:|---:|
| Baseline | Native LEGION objective | 42, 43, 44 | 3 |
| G | Baseline + explanation-state region BCE/Dice, coefficient 0.1 | 42, 43, 44 | 3 |
| P | G objective with reviewed auxiliary-gradient routing | 42, 43, 44 | 3 |
| **Total** | 3 arms × 3 seeds |  | **9** |

Use matched initialization per seed, the same data-order policy, 2,109 updates, effective batch 16, AdamW, bf16, world size 4, validation schedule, and fixed-final-update checkpoint as specified in `configs/experiment_matrix.yaml`. Before the nine runs:

- [ ] Implement P and verify with the real model that auxiliary gradients reach intended explanation-side parameters and do not reach the protected native-mask parameters.
- [ ] Verify zero auxiliary coefficient reproduces baseline outputs/gradients within a predeclared numerical tolerance; verify nonzero coefficient has the intended gradient path.
- [ ] Review a small engineering benchmark and estimate the 9-run wall time/storage from measurement. The historical G run adds SAM decoder calls and mask supervision, so do not claim equal compute.
- [ ] If the paper will claim gains at equal compute, design and run a compute-matched control (match measured FLOPs or accelerator time, define the control before seeing confirmatory test results). Otherwise state the additional compute plainly.
- [ ] Launch all planned arm/seed combinations through a manifest-driven runner; no best-seed or best-checkpoint selection on test data.

**Primary report:** per-seed localization metrics (foreground IoU/F1, mIoU, pixel precision/recall), ROUGE-L/CSS, empty and zero-overlap rates, calibration/abstention if implemented, wall time, decoder calls, and peak memory. Include paired image-level intervals within each seed and a separate seed-level summary; do not treat images from one checkpoint as independent training replications.

**Artifacts per run:** immutable config and logs, checkpoint pointer/hash, per-image text and masks, metric JSON/CSV, resource record, and failure record. Store large weights/data outside Git; provide paths or access instructions, not copied assets.

### Stage 2 — Independent-domain generalization (E2)

**Purpose:** check whether conclusions transfer beyond SynthScars.

- [ ] Select a licensed independent benchmark with the required image, source, explanation, and region labels (LOKI is a candidate, not yet a committed dataset).
- [ ] Verify source/image overlap against training and development data. Freeze source-level splits, inclusion/exclusion rules, prompts, thresholds, and evaluation code before inference.
- [ ] Run the frozen baseline, G, and P checkpoints from E1 on the complete eligible holdout. If SAE-Causal passes Stage 3, include it in the same frozen evaluation.
- [ ] Report source-clustered uncertainty, complete denominators, missing labels, subgroup performance, and all failed cases. No tuning on external holdout outputs.

**Deliverable:** independent-domain result table, source-disjointness audit, and prediction bundle keyed by anonymized example IDs. If no suitable licensed dataset is available, record the blocker and do not imply generalization.

### Stage 3 — SAE-Causal method line (B1–B3)

This line does not retrain the base model for the post-hoc arm. It still requires exact encoder compatibility and its own activation/feature artifacts. Keep images and extracted activations under the same access restrictions as the dataset.

**B1 — Encoder and dictionary gate**

- [ ] Identify and document the exact encoder revision, layer/tensor, patch ordering, normalization, and preprocessing used by the model.
- [ ] Test that the SAE hook disabled is numerically equivalent to the unmodified model; test round-trip patch indexing on synthetic and real batches.
- [ ] Fit dictionary parameters using training-split activations only. Use at least three dictionary initialization seeds for stability estimates; select width, sparsity penalty, and checkpoints using training/validation only.
- [ ] Report reconstruction error, activation sparsity/L0, dead-feature rate, feature stability across seeds, and resource cost. Freeze dictionary selection before held-out evaluation.

**B2 — Causal specificity gate**

- [ ] Bind each candidate feature to a claim token/span and patch support. Measure claim-token log-probability change when patching/ablating the feature.
- [ ] Compare against matched random features, matched activation-magnitude features, and region/pixel controls. Match interventions for affected activation mass where practical.
- [ ] Test whether claimed-region interventions have a larger, directionally correct effect than matched non-referent interventions. Include negative and null cases.
- [ ] Reject features that are unstable, reconstruct poorly, or have no specific causal effect. Do not select features using the final test set.

**B3 — Conditional gated method**

- [ ] Only after B1 and B2 pass, implement a claim-to-feature selector/gate with counterfactual consistency and an explicit abstention path.
- [ ] Train/select the selector on training/validation data only. Record its parameter count and show the base model remains frozen if that is the intended method.
- [ ] Compare post-hoc SAE-Causal with gated SAE-Causal on the same frozen examples. Verify that the gate improves a predeclared endpoint over post-hoc selection before running the full factorial study.

**Stop rule:** if the exact hook changes base behavior, the features are unstable, or causal effects do not beat matched controls, stop the gated branch. Preserve the negative result. Try a simpler sparse probe/dictionary only as a separately versioned method with fresh validation choices.

### Stage 4 — Factorial method comparison (E5, conditional)

Run only if E1 is complete and B1–B3 pass. Compare four cells on the same seeds and frozen data protocol:

| Cell | Spatial grounding | SAE-Causal |
|---|---|---|
| Native | No | No |
| G | Yes | No |
| SAE-Causal | No | Yes |
| G + SAE-Causal | Yes | Yes |

- [ ] First determine whether SAE-Causal is post-hoc or has a trained selector. Post-hoc evaluation reuses frozen E1 checkpoints; a trainable selector requires its own predeclared seeds and training jobs.
- [ ] Implement the combined cell without changing the frozen E1 method definitions. Record any interaction in compute, labels, or architecture.
- [ ] Use seeds 42/43/44 where training is repeated; report the exact number of additional jobs after the selector design is frozen. Do not silently count reused checkpoints as new training replications.
- [ ] Evaluate localization, causal faithfulness, text quality, calibration/abstention, and latency separately. Report the grounding × SAE interaction and its uncertainty.

### Stage 5 — Large-scale intervention study (E3)

**Purpose:** test whether explanations depend on the visual evidence they name.

- [ ] Pre-register a held-out set of eligible image-claim pairs and blinded label protocol. Determine the final sample size from available source clusters and a power/precision analysis before model inference; use an engineering pilot only to validate the pipeline.
- [ ] For each pair create the unchanged input, referent-region occlusion, and matched non-referent occlusion. Specify the occlusion algorithm and severity using development data, then freeze it.
- [ ] Run every frozen method on all three conditions. Score claim-specific token/log-probability changes, contradiction/abstention behavior, preservation of unrelated claims, patch-support IoU, and intervention margin.
- [ ] Have blinded annotators or an independent qualified process judge original support and whether the change is directionally appropriate. Keep method identity hidden during labeling.
- [ ] Use paired analysis and cluster intervals by source. Report the full eligible denominator and exclusions. Text similarity alone is not a faithfulness endpoint.

**Ready analysis scripts:** `analysis/intervention_analysis.py` accepts the annotated intervention CSV. Add schema validation and a documented example command before handoff.

### Stage 6 — Independent audit utility study (E4)

**Purpose:** test whether explanations help a person or independent auditor find unsupported claims.

- [ ] Assemble a separate, source-disjoint audit set with naturally occurring errors and carefully verified unsupported/fabricated claims. Define the error taxonomy and adjudication procedure before evaluation.
- [ ] Compare the same auditor/task with model decision and confidence only versus with the explanation evidence. Randomize condition order; blind method identity and keep labels hidden until scoring.
- [ ] Set sample size from a power/precision calculation using a pilot that is excluded from the final analysis. Freeze the false-alarm operating point on development data.
- [ ] Primary endpoint: error-discovery recall at the frozen false-alarm rate. Secondary endpoints: completion time, confidence calibration, unsupported-claim false reassurance, and error-type breakdown.
- [ ] Use source-clustered confidence intervals and report annotator agreement, adjudication, missing responses, and all exclusions.

**Ready analysis script:** `analysis/audit_utility.py` accepts blinded audit scores and development-frozen thresholds. This offline analysis is not a substitute for collecting the blinded audit data.

### Stage 7 — Robustness, ablations, and final analysis

Run only after primary definitions are frozen; mark these as secondary analyses.

- [ ] Stratify by scar/lesion size, location, image quality, and source family. Freeze bins on development data.
- [ ] Test background/context swaps, matched benign controls, and evidence-preserving transformations for shortcut reliance.
- [ ] Ablate region-loss weight, annotated-token coverage, and decoder capacity for G/P. Use validation to choose settings; reserve the final holdout for one evaluation.
- [ ] Compare SAE features with linear probes or simpler sparse dictionaries to test whether the added method complexity is useful.
- [ ] Report abstention coverage versus risk, calibration, error discovery, localization, causal specificity, and utility as distinct outcomes.
- [ ] Run paired bootstrap (10,000 resamples where configured), per-seed summaries, source-clustered external intervals, and failure analyses. Publish aggregate results and allowed per-example outputs without restricted images, activations, or weights.

## Handoff checklist for the student running jobs

- [ ] Start from a clean clone and follow `docs/REPRODUCE.md`; verify the pinned environment and assets before scheduling GPUs.
- [ ] Check the experiment matrix and protocol commit match the intended run. Never infer launchability from a planned row.
- [ ] Run preflight and submit a cost estimate based on measured throughput and storage. Current config expects four compatible GPUs for the training recipe.
- [ ] Use immutable run IDs; keep stdout/stderr, scheduler job ID, exit status, resource use, and checksums.
- [ ] Run the analysis from a clean checkout and compare output hashes/tables with the author run.
- [ ] Update `docs/EXPERIMENTS.md`, the claim ledger, and this file with status, commit, run IDs, results, and limitations. Failed or inconclusive results stay in the record.

## Next actions, in order

1. Portability + frozen source-level manifests (Stage 0).
2. Implement and review P gradient routing; validate the complete 9-job E1 matrix.
3. Launch E1 and analyze seeds separately.
4. Select/freeze the independent benchmark and run E2.
5. Run SAE encoder/dictionary and causal-specificity gates (B1/B2); stop if they fail.
6. Implement gated SAE-Causal and run the conditional factorial comparison only if justified.
7. Collect the blinded intervention and audit-utility datasets, with sample sizes fixed before the held-out analyses.
8. Finish secondary robustness/ablation work and write claims only to the level supported by all evidence.

## Ideas outside the committed experiment plan

- [ ] Evaluate whether human-readable feature labels are stable across dictionary seeds and harmless input transformations before assigning names.
- [ ] Test whether causally localized feature groups reveal omitted or contradictory visual evidence that region masks miss.
- [ ] Consider a reviewer interface showing claim, supporting patches, counterfactual effect, and uncertainty only after offline audit utility is established; evaluate it as a separate user study.
