# Experimental plan and claim boundaries

The full forward-looking work queue is in [`TODO.md`](../TODO.md). It breaks the work into gated stages, spells out the planned job counts and deliverables, and distinguishes ready analysis from methods that still need implementation. No planned large experiment below has been launched unless explicitly identified as existing evidence.

## Research question

Does making explanation representations spatially decodable improve artifact localization, and do the resulting explanations actually depend on the visual evidence they describe?

These are separate questions. A region loss can improve localization without making generated text faithful. Text similarity scores do not measure statement truth or faithfulness.

## Existing exploratory result: G

G trains with the original LEGION objective plus `0.1 * mean_ref(0.4 * BCE + 0.2 * Dice)` on masks aligned to complete official explanations. It uses the prediction-state span for each explanation, reuses the native image embedding, and adds one SAM prompt/mask-decoder call per eligible image. The auxiliary target is the official referent mask. It adds mask supervision and compute; it is not an equal-compute comparison.

Reported on one seed and 1,000 SynthScars test images:

| Metric | Baseline | G | Difference |
|---|---:|---:|---:|
| mIoU | 55.12 | 56.05 | +0.94 pp |
| Foreground F1 | 29.16 | 31.30 | +2.14 pp |
| ROUGE-L | 41.67 | 41.46 | -0.21 pp |
| CSS | 73.95 | 73.74 | -0.22 pp |

Paired image bootstrap intervals for localization were positive in the historical analysis. They capture image-sampling uncertainty only, not training-seed variation or selection after test feedback. The test set is exploratory for all subsequent method choices.

## Planned evaluation sequence

The first training batch is **3 full runs**: baseline, G, and P, each with seed 42. This count excludes smoke checks and engineering preflights. It is a single-seed comparison and does not estimate training-seed variability. P is not implemented yet, so this matrix is not launchable until its gradient gate passes. Later SAE, external-domain, intervention, and auditor studies are conditional work packages described in `TODO.md`; they have no results or fixed sample counts yet.

### E1. Fixed-seed comparison and gradient-routing check

Run baseline, G, and P from the same seed-42 initialization, with the same data order policy, update count, validation schedule, and fixed-final checkpoint. P is the proposed variant that routes auxiliary gradients to explanation-side parameters while preserving the native mask branch; it must pass a real-model gradient-routing check before any full run. Report added decoder calls, mask exposures, wall time, and peak memory. Do not call this equal-compute unless a measured compute-matched control is added.

Primary outcomes: foreground IoU/F1, mIoU, ROUGE-L, CSS, empty-mask count, zero-overlap count, pixel precision/recall, and per-image paired differences. Report paired image-level results for seed 42. Image-level intervals do not measure variation across training seeds.

### E2. Independent-domain evaluation

Freeze one final method and score it on an untouched external domain (for example LOKI or another licensed benchmark with region and explanation labels). Establish source/image disjointness before training and prompt selection. Freeze evaluation code and exclusions before inference. Report all failures and the complete planned denominator.

### E3. Explanation faithfulness by visual intervention

For each held-out image, bind an explanation claim to its annotated referent. Produce controlled inputs: unchanged image, referent-region occlusion, and a matched non-referent-region occlusion. Run the same frozen model and compare claim-specific outputs. A faithful dependency pattern should change more after changing the claimed evidence than after changing an unrelated region, while preserving the ability to abstain or report insufficient evidence.

The intervention analysis requires blinded human or independently qualified labels for whether the original claim is supported and whether the changed output is directionally appropriate. Similarity to the reference text alone is not a faithfulness label. Never use the official test images to tune the intervention, prompts, thresholds, or model.

The v2 implementation separates input-only validity from blinded output assertion. Its primary outcome is the joint pattern `a_intact * (1 - a_referent) * a_control`; selective assertion drop and all three component rates are secondary. Do not subtract two appropriate-response rates: an ideal model is correct in both conditions. The planned manifest fixes the denominator independently of observed rows, and source groups are equally weighted. See [`INTERVENTIONS.md`](INTERVENTIONS.md) for commands, masks, annotations, failure handling, and interpretation limits.

### E4. Independent audit usefulness

On a separately held-out set, introduce naturally occurring or carefully verified report errors. Compare an auditor with the explanation against the same auditor with the model's decision/confidence only. Primary outcome: error-discovery recall at a fixed false-alarm rate, with source-clustered confidence intervals. This experiment is required before claiming that explanations improve integrity auditing.

## Stop conditions

- Stop before the full run if the auxiliary gradient does not reach the intended trainable modules or leaks into a branch declared protected.
- Stop if the protocol changes after test outputs are inspected; create a new untouched test set for confirmatory claims.
- Retain failed jobs and report missing outcomes. No automatic retry, best-checkpoint selection by test score, or denominator reduction.
- If localization improves but intervention and audit outcomes do not, report a grounding/localization result only.

## Explicit nonclaims

The current G result does not establish explanation truth, causal faithfulness, source authenticity, calibrated confidence, independent audit value, generalization, or state of the art.
