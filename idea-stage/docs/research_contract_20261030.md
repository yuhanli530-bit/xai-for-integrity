# Research contract: XAI for Image Integrity

Updated 2026-10-30. This is a prospective contract for remaining work, not retrospective preregistration of the historical SynthScars pilot. It follows `docs/EXPERIMENTS.md`, `docs/METHOD_LINES.md`, and `TODO.md`. The skill's referenced contract template was unavailable locally; the project protocol supplies the requirements below.

## Objective and scope

Produce a paper with an implemented method, fair controls, reproducible training/evaluation, evidence-grounded analysis, and a clear account of limitations. Preserve both requested research lines:

1. **Spatial grounding:** G and proposed P connect explanation prediction states to annotated anomaly regions.
2. **Interpretable evidence mechanism:** select and implement SAE-Causal or a justified concept/token intervention alternative. Compare the selected mechanism against direct region/token and matched random-feature controls. If both lines pass independently, evaluate their combination as already planned in TODO.md.

SAE is a candidate tool, not a required positive result or proof of interpretability. A discarded candidate and its failure evidence remain in the development record. Novelty against current primary literature must be checked before making a paper contribution claim.

## Claims and required evidence

| Claim | Evidence required | Current status |
|---|---|---|
| C1: spatial grounding improves anomaly localization | Complete image-aligned baseline/G/P results; same initialization/data order/update and checkpoint policy; localization metrics and paired differences; actual compute and supervision accounting; correct-vs-shuffled alignment control | Historical G summary exists; exact G raw-score recovery/audit pending. P is unimplemented. |
| C2: explanations selectively depend on the evidence they name | Untouched source-disjoint claim manifest; input-only validity review; intact/referent/control outputs; blinded claim labels; joint evidence-pattern and component rates; paired source-level arm comparisons | V2 generation/analysis implemented and engineering-tested. No real held-out intervention result. |
| C3: internal interpretable features support a usable mechanism | Exact encoder/layer compatibility; reconstruction and activity checks if using SAE; meaningful features reviewed independently; targeted intervention vs matched random/dense or direct-token controls; held-out effects | Proposed; no implementation or empirical result. |
| C4: explanations help integrity auditing | Independent held-out reviewer experiment, explanation vs decision/confidence-only conditions, thresholds frozen on development data, error recall at fixed false-alarm rate, paired/source-aware uncertainty | Planned; no reviewer data. |
| C5: gains transfer beyond the development benchmark | Frozen external source-disjoint evaluation and appropriate real/unaltered negatives; report all outcomes/failures | Dataset/license/source verification pending. |

These claims must be decided separately. Better localization does not establish C2-C5. If evidence supports only C1, describe that result honestly and continue evaluating the requested second line; do not call the full paper objective completed by relabeling infrastructure as experiments.

## Fixed decisions

- Model training and any SAE dictionary training use one seed, **42**, as requested. No inference about training-seed variability.
- Planned baseline/G/P batch: four compatible GPUs; 2,109 updates; effective batch 16; AdamW, bf16; same data-order and final-checkpoint rules. P requires its real-model gradient gate first.
- The historical 1,000-image SynthScars test set informed development. It is exploratory, not untouched confirmation.
- Additional decoder calls, mask exposures, wall time, and memory must be measured. G has extra supervision/compute; do not claim equal-compute superiority without a corresponding control.
- Use actual licensed dataset annotations and qualified human labels. Model-generated interpretations are predictions, not ground truth.
- No automatic full-job launch from a passed preflight; observe the repository's allocation and finite-launch policy. Do not restart/retrain an existing experiment because a read-only status query timed out.

## E3 measurement decisions

The primary per-claim outcome is `aI * (1-aR) * aC`, with a fixed original claim and assertion labels on intact, referent-occluded, and control-occluded inputs. Eligibility is input-only support `(true,false,true)` and identical across arms. Report all component rates and selective assertion drop `aC-aR`; the latter alone is insufficient. Use equal source weighting, keeping every triplet together. See `docs/INTERVENTIONS.md` for executable commands and schemas.

Preserve all planned claims including construction failures and semantically invalid interventions. Missing rows/labels cannot produce complete estimates. Never tune masks, claims, prompts, eligibility, or method selection after inspecting held-out output behavior. Source-sampling confidence intervals are conditional on the frozen model and reviewed labels.

## Order of remaining work

1. Recover and recompute the historical per-image G comparison; inventory retained H/S and other development candidates. Keep positive and negative evidence, and identify different protocols rather than pooling them.
2. Validate the native model/data stack and implement P. Complete zero-coefficient equivalence, gradient-routing, and short training checks before the full comparison.
3. Implement a frozen-model triplet inference adapter and a small development-only annotation pilot. Use it to determine the source count and annotation resources, then freeze the large E3 manifest and protocol before inference. The large sample count is not yet fixed; no full E3 run is ready.
4. Select/implement Line B based on feasibility and novelty evidence. Evaluate the selected method and requested combination if feasible, with the specified controls.
5. Complete external-domain/negative-control and independent reviewer studies. Review the audit-utility script's missing-pair and weighting behavior before using it for C4.
6. Assemble paper tables/figures from raw outputs, draft the manuscript, verify citations, audit every quantitative claim, and conduct a full-method review. A compiled manuscript alone is not completion.

## Completion evidence

Completion requires verified code and exact run commands, preserved complete run manifests and results, independently reviewed claims, the requested two-line investigation, relevant controls and failures, and a coherent manuscript with traceable tables/figures and references. A paper may report a negative or limited result; it may not substitute unrun plans, CPU fixtures, model access, or a polished progress report for scientific evidence.
